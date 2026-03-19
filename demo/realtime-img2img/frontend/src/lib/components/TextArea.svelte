<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value: string;
  export let params: FieldProps;
  onMount(() => {
    value = String(params?.default ?? '');
  });
</script>

<div>
  <div class="mb-1 flex items-center gap-1">
    <label class="text-sm font-medium" for={params?.id}>
      {params?.title}
    </label>
    {#if params?.description}
      <div class="group relative inline-block">
        <span class="cursor-help select-none text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-200">ⓘ</span>
        <div class="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1 w-64 -translate-x-1/2 rounded bg-gray-800 p-2 text-xs text-white opacity-0 shadow-lg transition-opacity group-hover:opacity-100">
          {params.description}
        </div>
      </div>
    {/if}
  </div>
  <div class="text-normal flex items-center rounded-md border border-gray-700">
    <textarea
      class="mx-1 w-full px-3 py-2 font-light outline-none dark:text-black"
      id={params?.id}
      title={params?.title}
      placeholder="Add your prompt here..."
      bind:value
    ></textarea>
  </div>
</div>
