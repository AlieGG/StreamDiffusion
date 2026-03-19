from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request

import markdown2

import logging
import uuid
import time
import threading
from types import SimpleNamespace
import asyncio
import os
import time
import mimetypes
import torch
from huggingface_hub import snapshot_download

from config import config, Args
from util import pil_to_frame, bytes_to_pil
from connection_manager import ConnectionManager, ServerFullException
from img2img import Pipeline, AVAILABLE_MODELS

# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")

THROTTLE = 1.0 / 120
# logging.basicConfig(level=logging.DEBUG)

_download_state = {"downloading": None, "error": None, "completed": []}
_download_lock = threading.Lock()


def _check_model_cached(model_id: str) -> bool:
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = "models--" + model_id.replace("/", "--")
    snapshots_dir = os.path.join(cache_dir, model_dir, "snapshots")
    return os.path.exists(snapshots_dir) and bool(os.listdir(snapshots_dir))


def _download_model_thread(model_id: str):
    try:
        snapshot_download(
            repo_id=model_id,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*", "rust_model*"],
        )
        with _download_lock:
            if model_id not in _download_state["completed"]:
                _download_state["completed"].append(model_id)
    except Exception as e:
        with _download_lock:
            _download_state["error"] = str(e)
    finally:
        with _download_lock:
            _download_state["downloading"] = None


class App:
    def __init__(self, config: Args, pipeline):
        self.args = config
        self.pipeline = pipeline
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        self.init_app()

    def init_app(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
            try:
                await self.conn_manager.connect(
                    user_id, websocket, self.args.max_queue_size
                )
                await handle_websocket_data(user_id)
            except ServerFullException as e:
                logging.error(f"Server Full: {e}")
            finally:
                await self.conn_manager.disconnect(user_id)
                logging.info(f"User disconnected: {user_id}")

        async def handle_websocket_data(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                return HTTPException(status_code=404, detail="User not found")
            last_time = time.time()
            try:
                while True:
                    if (
                        self.args.timeout > 0
                        and time.time() - last_time > self.args.timeout
                    ):
                        await self.conn_manager.send_json(
                            user_id,
                            {
                                "status": "timeout",
                                "message": "Your session has ended",
                            },
                        )
                        await self.conn_manager.disconnect(user_id)
                        return
                    data = await self.conn_manager.receive_json(user_id)
                    if data["status"] == "next_frame":
                        info = pipeline.Info()
                        params = await self.conn_manager.receive_json(user_id)
                        params = pipeline.InputParams(**params)
                        params = SimpleNamespace(**params.dict())
                        if info.input_mode == "image":
                            image_data = await self.conn_manager.receive_bytes(user_id)
                            if len(image_data) == 0:
                                await self.conn_manager.send_json(
                                    user_id, {"status": "send_frame"}
                                )
                                continue
                            params.image = bytes_to_pil(image_data)
                        await self.conn_manager.update_data(user_id, params)

            except Exception as e:
                logging.error(f"Websocket Error: {e}, {user_id} ")
                await self.conn_manager.disconnect(user_id)

        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        @self.app.get("/api/ready")
        async def ready():
            return JSONResponse({"ready": pipeline.is_ready})

        @self.app.get("/api/stream/output")
        async def stream_output(request: Request):
            async def generate():
                while True:
                    if await request.is_disconnected():
                        break
                    frame = pipeline.last_frame
                    if frame is not None:
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            + f"Content-Length: {len(frame)}\r\n\r\n".encode()
                            + frame
                            + b"\r\n"
                        )
                    await asyncio.sleep(1 / 60)

            return StreamingResponse(
                generate(),
                media_type="multipart/x-mixed-replace;boundary=frame",
                headers={"Cache-Control": "no-cache"},
            )

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            try:

                async def generate():
                    loop = asyncio.get_event_loop()
                    while True:
                        last_time = time.time()
                        await self.conn_manager.send_json(
                            user_id, {"status": "send_frame"}
                        )
                        params = await self.conn_manager.get_latest_data(user_id)
                        if params is None:
                            continue
                        try:
                            image = await loop.run_in_executor(None, pipeline.predict, params)
                        except Exception as e:
                            logging.error(f"predict executor error: {e}", exc_info=True)
                            continue
                        if image is None:
                            continue
                        frame = pil_to_frame(image)
                        yield frame
                        if self.args.debug:
                            print(f"Time taken: {time.time() - last_time}")

                return StreamingResponse(
                    generate(),
                    media_type="multipart/x-mixed-replace;boundary=frame",
                    headers={"Cache-Control": "no-cache"},
                )
            except Exception as e:
                logging.error(f"Streaming Error: {e}, {user_id} ")
                return HTTPException(status_code=404, detail="User not found")

        # route to setup frontend
        @self.app.get("/api/settings")
        async def settings():
            info_schema = pipeline.Info.schema()
            info = pipeline.Info()
            if info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = pipeline.InputParams.schema()
            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "max_queue_size": self.args.max_queue_size,
                    "page_content": page_content if info.page_content else "",
                }
            )

        @self.app.get("/api/models")
        async def list_models():
            models = []
            with _download_lock:
                currently_downloading = _download_state["downloading"]
            for model_id in AVAILABLE_MODELS:
                models.append(
                    {
                        "id": model_id,
                        "name": model_id.split("/")[-1],
                        "downloaded": _check_model_cached(model_id),
                        "downloading": currently_downloading == model_id,
                    }
                )
            return JSONResponse({"models": models})

        @self.app.post("/api/models/download")
        async def download_model(request: Request):
            data = await request.json()
            model_id = data.get("model_id")
            if not model_id or model_id not in AVAILABLE_MODELS:
                raise HTTPException(status_code=400, detail="Invalid model ID")
            with _download_lock:
                if _download_state["downloading"]:
                    raise HTTPException(
                        status_code=409, detail="Another download is in progress"
                    )
                _download_state["downloading"] = model_id
                _download_state["error"] = None
            t = threading.Thread(
                target=_download_model_thread, args=(model_id,), daemon=True
            )
            t.start()
            return JSONResponse({"status": "started"})

        @self.app.get("/api/models/download-status")
        async def download_status():
            with _download_lock:
                return JSONResponse(dict(_download_state))

        if not os.path.exists("public"):
            os.makedirs("public")

        self.app.mount(
            "/", StaticFiles(directory="./frontend/public", html=True), name="public"
        )


device = torch.device("cuda" if torch.cuda.is_available() else "mps")
torch_dtype = torch.float16
pipeline = Pipeline(config, device, torch_dtype)
app = App(config, pipeline).app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        ssl_certfile=config.ssl_certfile,
        ssl_keyfile=config.ssl_keyfile,
    )
