import { matMul, matMulGPU, randomMatrix, toFlatArray } from "./matMul/matMul";

const m1RowsElem = document.getElementById('first-matrix-rows') as HTMLInputElement;
const m1ColsElem = document.getElementById('first-matrix-cols') as HTMLInputElement;

const m2RowsElem = document.getElementById('second-matrix-rows') as HTMLInputElement;
const m2ColsElem = document.getElementById('second-matrix-cols') as HTMLInputElement;

const cpuTimeElem = document.getElementById('cpu-time') as HTMLSpanElement;
const gpuTimeElem = document.getElementById('gpu-time') as HTMLSpanElement;
const gpuTimeTiledElem = document.getElementById('gpu-time-tiled') as HTMLSpanElement;

const computeBtn = document.getElementById('compute-btn') as HTMLButtonElement;
computeBtn.addEventListener('click', () => { compute() });

// m2 rows = m1 cols
m1ColsElem.addEventListener('input', () => {
    m2RowsElem.value = m1ColsElem.value;
});

const timeUnit = (ms: number): string => {
    let unit = 'ms';
    if (ms >= 1000) {
        ms /= 1000;
        unit = 's';
    }

    return `${ms.toFixed(3)}${unit}`
}

function logCompare(m1: Float32Array, m2: Float32Array) {
    let noDiffs = true;
    for (let i = 0; i < m1.length; ++i) {
        if (m1[i] !== m2[i]) {
            console.log('diff at elem ' + i + ' ' + m1[i] + ' ' + m2[i]);
            noDiffs = false;
        }
    }
    if (noDiffs) {
        console.log('no diffs in matrices');
    }
}

async function compute() {
    const m1Rows = parseInt(m1RowsElem.value);
    const m1Cols = parseInt(m1ColsElem.value);
    const m2Cols = parseInt(m2ColsElem.value);

    const m1 = randomMatrix(m1Rows, m1Cols);
    const m2 = randomMatrix(m1Cols, m2Cols);

    const m1Flat = toFlatArray(m1);
    const m2Flat = toFlatArray(m2);

    let elapsedTime = 0;
    let result: number[][] | Float32Array;
    ({ elapsedTime, result } = matMul(m1, m2));
    const cpuTime = elapsedTime;
    cpuTimeElem.innerText = `${timeUnit(elapsedTime)}`;
    console.log(result);

    ({ elapsedTime, result } = await matMulGPU(m1Flat, m2Flat, false));
    const gpuTime = elapsedTime
    gpuTimeElem.innerText = `${timeUnit(elapsedTime)}  -  ${(cpuTime / gpuTime).toFixed(2)} speedup`;
    console.log(result);

    const previousResult = result;
    ({ elapsedTime, result } = await matMulGPU(m1Flat, m2Flat, true));
    logCompare(previousResult, result);
    const gpuTimeTiled = elapsedTime;
    gpuTimeTiledElem.innerText = `${timeUnit(elapsedTime)} - ${(cpuTime / gpuTimeTiled).toFixed(2)} speedup`;
    console.log(result);
}