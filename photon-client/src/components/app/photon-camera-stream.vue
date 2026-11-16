<script setup lang="ts">
import { computed, inject, onBeforeUnmount, onMounted, ref, useTemplateRef, watch } from "vue";
import { useStateStore } from "@/stores/StateStore";
import { useCameraSettingsStore } from "@/stores/settings/CameraSettingsStore";
import type { StyleValue } from "vue";
import PvIcon from "@/components/common/pv-icon.vue";
import type { UiCameraConfiguration } from "@/types/SettingTypes";
import PvLoading from "@/components/common/pv-loading.vue";

const props = defineProps<{
  streamType: "Raw" | "Processed";
  id: string;
  cameraSettings: UiCameraConfiguration;
}>();

const backendHostname = inject<string>("backendHostname");
const emptyStreamSrc = "//:0";
const demandId = `stream-${props.id}-${Math.random().toString(36).slice(2)}`;
const previewEnabled = ref(false);
const streamDemandActive = ref(false);
const effectiveStreamType = computed<"Raw" | "Processed">(() => {
  const pipelineSettings = props.cameraSettings.pipelineSettings;

  if (props.streamType === "Raw" && !pipelineSettings.inputShouldShow && pipelineSettings.outputShouldShow) {
    return "Processed";
  }

  return props.streamType;
});
const streamSrc = computed<string>(() => {
  const port = props.cameraSettings.stream[effectiveStreamType.value === "Raw" ? "inputPort" : "outputPort"];

  if (!useStateStore().backendConnected || port === 0) {
    return emptyStreamSrc;
  }

  return `http://${backendHostname}:${port}/stream.mjpg`;
});
const displayedStreamSrc = computed<string>(() => (streamDemandActive.value ? streamSrc.value : emptyStreamSrc));
const streamDesc = computed<string>(() => `${effectiveStreamType.value} Stream View`);
const previewKey = computed<string>(() => `${props.cameraSettings.uniqueName}:${effectiveStreamType.value}:${props.id}`);
const streamStyle = computed<StyleValue>(() => {
  if (useStateStore().colorPickingMode) {
    return { cursor: "crosshair" };
  }

  return {};
});

const containerStyle = computed<StyleValue>(() => {
  if (props.cameraSettings.validVideoFormats.length === 0) {
    return { aspectRatio: "1/1" };
  }
  const resolution =
    props.cameraSettings.validVideoFormats[props.cameraSettings.pipelineSettings.cameraVideoModeIndex].resolution;
  const rotation = props.cameraSettings.pipelineSettings.inputImageRotationMode;
  if (rotation === 1 || rotation === 3) {
    return {
      aspectRatio: `${resolution.height}/${resolution.width}`
    };
  }
  return {
    aspectRatio: `${resolution.width}/${resolution.height}`
  };
});

const overlayStyle = computed<StyleValue>(() => {
  if (useStateStore().colorPickingMode || displayedStreamSrc.value === emptyStreamSrc) {
    return { display: "none" };
  } else {
    return {};
  }
});

const handleCaptureClick = () => {
  if (effectiveStreamType.value === "Raw") {
    useCameraSettingsStore().saveInputSnapshot();
  } else {
    useCameraSettingsStore().saveOutputSnapshot();
  }
};
const handlePopoutClick = () => {
  if (!streamDemandActive.value) return;
  window.open(displayedStreamSrc.value);
};
const handlePreviewToggle = () => {
  const nextPreviewEnabled = !previewEnabled.value;
  previewEnabled.value = nextPreviewEnabled;
  useStateStore().setActiveStreamPreviewKey(nextPreviewEnabled ? previewKey.value : null);
  publishStreamDemand();
};
const handleFullscreenRequest = async () => {
  if (!streamDemandActive.value) return;
  const stream = document.getElementById(props.id);
  if (!stream) return;
  await stream.requestFullscreen();
};

const mjpgStream = useTemplateRef("mjpgStream");

const handleStreamError = () => {
  if (displayedStreamSrc.value && displayedStreamSrc.value !== emptyStreamSrc) {
    console.error("Error loading stream:", displayedStreamSrc.value, " Trying again.");
    setTimeout(() => {
      if (mjpgStream.value && streamDemandActive.value) {
        mjpgStream.value.src = displayedStreamSrc.value;
      }
    }, 100);
  }
};

const shouldDemandStream = () =>
  previewEnabled.value &&
  useStateStore().backendConnected &&
  streamSrc.value !== emptyStreamSrc &&
  document.visibilityState === "visible" &&
  document.hasFocus();

const publishStreamDemand = () => {
  const active = shouldDemandStream();
  streamDemandActive.value = active;
  useCameraSettingsStore().setStreamDemand(
    props.cameraSettings.uniqueName,
    demandId,
    active && effectiveStreamType.value === "Raw",
    active && effectiveStreamType.value === "Processed"
  );
};

let demandHeartbeat: number | undefined;

onMounted(() => {
  document.addEventListener("visibilitychange", publishStreamDemand);
  window.addEventListener("focus", publishStreamDemand);
  window.addEventListener("blur", publishStreamDemand);
  demandHeartbeat = window.setInterval(publishStreamDemand, 1000);
  publishStreamDemand();
});

watch([streamSrc, effectiveStreamType], publishStreamDemand);
watch(
  () => useStateStore().activeStreamPreviewKey,
  (activePreviewKey) => {
    if (activePreviewKey === previewKey.value || !previewEnabled.value) return;
    previewEnabled.value = false;
    publishStreamDemand();
  }
);

onBeforeUnmount(() => {
  if (demandHeartbeat !== undefined) {
    window.clearInterval(demandHeartbeat);
  }
  document.removeEventListener("visibilitychange", publishStreamDemand);
  window.removeEventListener("focus", publishStreamDemand);
  window.removeEventListener("blur", publishStreamDemand);
  if (useStateStore().activeStreamPreviewKey === previewKey.value) {
    useStateStore().setActiveStreamPreviewKey(null);
  }
  useCameraSettingsStore().setStreamDemand(props.cameraSettings.uniqueName, demandId, false, false);
  streamDemandActive.value = false;
  if (!mjpgStream.value) return;
  mjpgStream.value.src = emptyStreamSrc;
});
</script>

<template>
  <div class="stream-container" :style="containerStyle">
    <pv-loading v-if="streamDemandActive" class="stream-loading" />
    <img
      :key="displayedStreamSrc"
      :id="id"
      ref="mjpgStream"
      class="stream-video"
      crossorigin="anonymous"
      :src="displayedStreamSrc"
      :alt="streamDesc"
      :style="streamStyle"
      @error="handleStreamError"
    />
    <div v-if="!streamDemandActive" class="stream-idle">
      <v-btn
        icon
        color="primary"
        variant="tonal"
        :aria-label="`Enable ${streamDesc}`"
        :title="`Enable ${streamDesc}`"
        @click="handlePreviewToggle"
      >
        <v-icon>mdi-eye</v-icon>
      </v-btn>
    </div>
    <div class="stream-overlay" :style="overlayStyle">
      <pv-icon
        color="primary"
        icon-name="mdi-eye-off"
        tooltip="Disable this stream"
        class="ma-1 mr-2"
        @click="handlePreviewToggle"
      />
      <pv-icon
        color="primary"
        icon-name="mdi-camera-image"
        tooltip="Capture and save a frame of this stream"
        class="ma-1 mr-2"
        @click="handleCaptureClick"
      />
      <pv-icon
        color="primary"
        icon-name="mdi-fullscreen"
        tooltip="Open this stream in fullscreen"
        class="ma-1 mr-2"
        @click="handleFullscreenRequest"
      />
      <pv-icon
        color="primary"
        icon-name="mdi-open-in-new"
        tooltip="Open this stream in a new window"
        class="ma-1 mr-2"
        @click="handlePopoutClick"
      />
    </div>
  </div>
</template>

<style scoped>
.stream-container {
  display: flex;
  position: relative;
  width: 100%;
  height: 100%;
  max-width: 100%;
  max-height: 100%;
  justify-content: center;
  align-items: center;
}

.stream-loading {
  position: absolute;
  width: 25%;
  height: 25%;
  object-fit: contain;
}

.stream-video {
  position: absolute;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.stream-idle {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.stream-overlay {
  display: flex;
  opacity: 0;
  transition: 0.1s ease;
  position: absolute;
  top: 0;
  right: 0;
}

.stream-container:hover .stream-overlay {
  opacity: 1;
}
</style>
