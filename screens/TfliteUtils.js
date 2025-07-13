import RNFS from 'react-native-fs';
import { loadTensorflowModel} from 'react-native-fast-tflite';
import { Buffer } from 'buffer';
import jpeg from 'jpeg-js';


let model = null;
export async function loadModelIfNeeded() {
  if (!model) {
    model = await loadTensorflowModel(require('../assets/anpr2_yolov9_int8.tflite'));
  }
  return model;
}

export async function runInferenceOnImage(path) {
  const m = await loadModelIfNeeded();
  const b64 = await RNFS.readFile(path, 'base64');
  const tensor = base64ToTensor(b64, 640, 640);
  const output = await m.run([tensor]);
  return output;
}

export function base64ToTensor(base64, targetW, targetH) {
  const b64 = base64.replace(/^data:image\/\w+;base64,/, '');
  const imageBuffer = Buffer.from(b64, 'base64');
  const raw = jpeg.decode(imageBuffer, { useTArray: true });

  const resized = resizeNearestNeighbor(raw, targetW, targetH);

  const { data: rgba } = resized;
  const tensor = new Float32Array(targetW * targetH * 3);
  for (let i = 0; i < targetW * targetH; i++) {
    tensor[i * 3 + 0] = rgba[i * 4 + 0] / 255.0;
    tensor[i * 3 + 1] = rgba[i * 4 + 1] / 255.0;
    tensor[i * 3 + 2] = rgba[i * 4 + 2] / 255.0;
  }
  return tensor;
}


function resizeNearestNeighbor(img, newW, newH) {
  const { width: oldW, height: oldH, data: oldData } = img;
  const newData = new Uint8Array(newW * newH * 4);

  for (let y = 0; y < newH; y++) {
    const srcY = Math.floor(y * oldH / newH);
    for (let x = 0; x < newW; x++) {
      const srcX = Math.floor(x * oldW / newW);
      const srcIdx = (srcY * oldW + srcX) * 4;
      const dstIdx = (y * newW + x) * 4;
      // copy R,G,B,A
      newData[dstIdx + 0] = oldData[srcIdx + 0];
      newData[dstIdx + 1] = oldData[srcIdx + 1];
      newData[dstIdx + 2] = oldData[srcIdx + 2];
      newData[dstIdx + 3] = oldData[srcIdx + 3];
    }
  }

  return { width: newW, height: newH, data: newData };
}