<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value = 8.0;
  export let params: FieldProps;
  onMount(() => {
    value = Number(params?.default) ?? 8.0;
  });
</script>

<div class="grid max-w-md grid-cols-4 items-center gap-3">
  <div class="col-span-1 flex items-center gap-1">
    <label class="text-sm font-medium" for={params.id}>{params?.title}</label>
    {#if params?.description}
      <div class="group relative inline-block">
        <span class="cursor-help select-none text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">ⓘ</span>
        <div class="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1 w-52 -translate-x-1/2 rounded bg-gray-800 p-2 text-xs text-white opacity-0 shadow-lg transition-opacity group-hover:opacity-100">
          {params.description}
        </div>
      </div>
    {/if}
  </div>
  <input
    class="col-span-2 h-2 w-full cursor-pointer appearance-none rounded-lg bg-gray-300 dark:bg-gray-500"
    bind:value
    type="range"
    id={params.id}
    name={params.id}
    min={params?.min}
    max={params?.max}
    step={params?.step ?? 1}
  />
  <input
    type="number"
    step={params?.step ?? 1}
    bind:value
    class="rounded-md border px-1 py-1 text-center text-xs font-bold dark:text-black"
  />
</div>
