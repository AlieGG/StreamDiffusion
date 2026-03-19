<script lang="ts">
  import { lcmLiveStatus, LCMLiveStatus, streamId } from '$lib/lcmLive';
  import { getPipelineValues } from '$lib/store';

  import Button from '$lib/components/Button.svelte';
  import Floppy from '$lib/icons/floppy.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import { snapImage } from '$lib/utils';
  import { onDestroy } from 'svelte';

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;
  $: console.log('isLCMRunning', isLCMRunning);
  let imageEl: HTMLImageElement;
  let isReady = true;
  let pollInterval: ReturnType<typeof setInterval> | undefined;

  $: if (isLCMRunning) {
    startPolling();
  } else {
    stopPolling();
    isReady = true;
  }

  function startPolling() {
    if (pollInterval) return;
    pollInterval = setInterval(async () => {
      try {
        const data = await fetch('/api/ready').then((r) => r.json());
        isReady = data.ready;
      } catch {
        // server busy, keep last known state
      }
    }, 500);
  }

  function stopPolling() {
    if (pollInterval) {
      clearInterval(pollInterval);
      pollInterval = undefined;
    }
  }

  onDestroy(stopPolling);

  async function takeSnapshot() {
    if (isLCMRunning) {
      await snapImage(imageEl, {
        prompt: getPipelineValues()?.prompt,
        negative_prompt: getPipelineValues()?.negative_prompt,
        seed: getPipelineValues()?.seed,
        guidance_scale: getPipelineValues()?.guidance_scale
      });
    }
  }
</script>

<div
  class="relative mx-auto aspect-square max-w-lg self-center overflow-hidden rounded-lg border border-slate-300"
>
  <!-- svelte-ignore a11y-missing-attribute -->
  {#if isLCMRunning && $streamId}
    <img
      bind:this={imageEl}
      class="aspect-square w-full rounded-lg"
      src={'/api/stream/' + $streamId}
    />
    {#if !isReady}
      <div class="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-black bg-opacity-60 text-white">
        <Spinner classList={'animate-spin w-10 h-10 opacity-90'} />
        <p class="text-sm font-medium">Reloading pipeline…</p>
      </div>
    {/if}
    <div class="absolute bottom-1 right-1">
      <Button
        on:click={takeSnapshot}
        disabled={!isLCMRunning}
        title={'Take Snapshot'}
        classList={'text-sm ml-auto text-white p-1 shadow-lg rounded-lg opacity-50'}
      >
        <Floppy classList={''} />
      </Button>
    </div>
  {:else}
    <img
      class="aspect-square w-full rounded-lg"
      src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    />
  {/if}
</div>
