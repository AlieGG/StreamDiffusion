<script lang="ts">
  import type { FieldProps } from '$lib/types';
  import { onMount } from 'svelte';
  export let value = false;
  export let params: FieldProps;
  onMount(() => {
    value = Boolean(params?.default) ?? false;
  });
</script>

<div class="grid max-w-md grid-cols-4 items-center justify-items-start gap-3">
  <div class="col-span-3 flex items-center gap-1">
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
  <input bind:checked={value} type="checkbox" id={params.id} class="cursor-pointer" />
</div>
