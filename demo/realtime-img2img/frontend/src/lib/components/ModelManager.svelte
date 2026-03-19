<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Spinner from '$lib/icons/spinner.svelte';

  type Model = {
    id: string;
    name: string;
    downloaded: boolean;
    downloading: boolean;
  };

  let models: Model[] = [];
  let error = '';
  let pollInterval: ReturnType<typeof setInterval> | undefined;

  async function fetchModels() {
    const data = await fetch('/api/models').then((r) => r.json());
    models = data.models;
  }

  async function downloadModel(modelId: string) {
    error = '';
    const res = await fetch('/api/models/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: modelId })
    });
    if (!res.ok) {
      const data = await res.json();
      error = data.detail || 'Failed to start download';
      return;
    }
    await fetchModels();
  }

  onMount(() => {
    fetchModels();
    pollInterval = setInterval(async () => {
      const data = await fetch('/api/models/download-status').then((r) => r.json());
      if (data.downloading !== null || data.error) {
        await fetchModels();
        if (data.error) error = data.error;
      }
    }, 2000);
  });

  onDestroy(() => {
    if (pollInterval) clearInterval(pollInterval);
  });
</script>

<div class="flex flex-col gap-4 py-4">
  <div>
    <h2 class="text-xl font-bold">Model Manager</h2>
    <p class="mt-1 text-sm text-gray-500 dark:text-gray-400">
      Pre-download models here. Downloaded models switch instantly in the stream view.
    </p>
  </div>

  {#if error}
    <p class="rounded-md bg-red-100 px-3 py-2 text-sm text-red-700 dark:bg-red-900 dark:text-red-200">
      {error}
    </p>
  {/if}

  <div class="flex flex-col gap-3">
    {#each models as model}
      <div
        class="flex items-center justify-between rounded-lg border border-slate-300 p-4 dark:border-slate-700"
      >
        <div>
          <p class="font-medium">{model.name}</p>
          <p class="text-xs text-gray-500 dark:text-gray-400">{model.id}</p>
        </div>
        <div class="flex items-center gap-3">
          {#if model.downloading}
            <span class="flex items-center gap-2 text-sm text-blue-500">
              <Spinner classList="animate-spin w-4 h-4" />
              Downloading…
            </span>
          {:else if model.downloaded}
            <span class="text-sm font-medium text-green-600 dark:text-green-400">
              ✓ Ready
            </span>
          {:else}
            <button
              class="rounded-md bg-blue-600 px-3 py-1.5 text-sm text-white hover:bg-blue-700 active:bg-blue-800"
              on:click={() => downloadModel(model.id)}
            >
              Download
            </button>
          {/if}
        </div>
      </div>
    {/each}
  </div>
</div>
