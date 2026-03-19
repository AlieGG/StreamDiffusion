<script lang="ts">
  import { onMount, onDestroy } from 'svelte';

  let isFullscreen = false;
  let showControls = true;
  let hideTimer: ReturnType<typeof setTimeout> | undefined;
  let streamSrc = '';

  function scheduleHide() {
    clearTimeout(hideTimer);
    showControls = true;
    hideTimer = setTimeout(() => {
      showControls = false;
    }, 3000);
  }

  function toggleFullscreen() {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  }

  onMount(() => {
    streamSrc = '/api/stream/output';
    document.addEventListener('fullscreenchange', () => {
      isFullscreen = !!document.fullscreenElement;
    });
    scheduleHide();
  });

  onDestroy(() => {
    clearTimeout(hideTimer);
  });
</script>

<svelte:head>
  <title>Stream Viewer</title>
</svelte:head>

<!-- svelte-ignore a11y-no-static-element-interactions -->
<div
  class="relative flex h-screen w-screen items-center justify-center overflow-hidden bg-black"
  on:mousemove={scheduleHide}
  on:click={scheduleHide}
  style="cursor: {showControls ? 'default' : 'none'}"
>
  {#if streamSrc}
    <img
      src={streamSrc}
      alt="AI Stream"
      class="max-h-screen max-w-full object-contain"
    />
  {/if}

  <!-- Floating controls -->
  <div
    class="absolute right-4 top-4 flex items-center gap-2 transition-opacity duration-500"
    style="opacity: {showControls ? 1 : 0}; pointer-events: {showControls ? 'auto' : 'none'}"
  >
    <a
      href="/"
      class="rounded-lg bg-black bg-opacity-60 px-3 py-2 text-sm text-white hover:bg-opacity-80"
    >
      ← Controls
    </a>
    <button
      on:click={toggleFullscreen}
      class="rounded-lg bg-black bg-opacity-60 px-3 py-2 text-sm text-white hover:bg-opacity-80"
      title={isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'}
    >
      {#if isFullscreen}
        ⛶ Exit Fullscreen
      {:else}
        ⛶ Fullscreen
      {/if}
    </button>
  </div>
</div>

<style>
  :global(body) {
    margin: 0;
    padding: 0;
    background: black;
    overflow: hidden;
  }
</style>
