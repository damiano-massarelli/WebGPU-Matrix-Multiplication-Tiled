import { default as matMulShader } from './matMul.wgsl';
import { default as matMulShaderTiled } from './matMul_tiled.wgsl';

export function randomMatrix(rows: number, cols: number) {
    const matrix = [];
    for (let i = 0; i < rows; ++i) {
        matrix.push(Array.from({ length: cols }, () => Math.random() * 10));
    }

    return matrix;
}

// row major linearization
export function toFlatArray(matrix: number[][]): Float32Array {
    const flatArray = [];
    for (let row of matrix) {
        flatArray.push(...row);
    }

    return new Float32Array([matrix.length, matrix[0].length, ...flatArray]);
}

// cpu version
export function matMul(firstMatrix: number[][], secondMatrix: number[][]) {
    const startTime = performance.now();

    const result: number[][] = [];
    for (let i = 0; i < firstMatrix.length; ++i) {
        const current: number[] = [];
        result.push(current);
        for (let j = 0; j < secondMatrix[0].length; ++j) {
            let accumulator = 0;
            for (let k = 0; k < firstMatrix[0].length; ++k) {
                accumulator += firstMatrix[i][k] * secondMatrix[k][j];
            }
            current.push(accumulator);
        }
    }

    const elapsedTime = performance.now() - startTime;

    return {
        result,
        elapsedTime,
    };
}

// gpu version, both tiled and naive
export async function matMulGPU(firstMatrix: Float32Array, secondMatrix: Float32Array, tiled: boolean) {
    if (firstMatrix.length < 2 || secondMatrix.length < 2) {
        throw new Error("Input matrix must contain at least the first two elements specifying its size.");
    }

    const adapter = await navigator.gpu.requestAdapter();

    if (!adapter) { // WebGPU not available
        return {
            result: new Float32Array(0),
            elapsedTime: -1
        };
    }

    const device = await adapter.requestDevice();

    const firstMatrixBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: firstMatrix.byteLength,
        usage: GPUBufferUsage.STORAGE
    });
    new Float32Array(firstMatrixBuffer.getMappedRange()).set(firstMatrix);
    firstMatrixBuffer.unmap();

    const secondMatrixBuffer = device.createBuffer({
        mappedAtCreation: true,
        size: secondMatrix.byteLength,
        usage: GPUBufferUsage.STORAGE
    });
    new Float32Array(secondMatrixBuffer.getMappedRange()).set(secondMatrix);
    secondMatrixBuffer.unmap();

    // First and second element of the matrix array contain its rows and columns
    const resultSizeByte = (2 + firstMatrix[0] * secondMatrix[1]) * Float32Array.BYTES_PER_ELEMENT; // 2 for the size
    const resultMatrixBuffer = device.createBuffer({
        size: resultSizeByte,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    });

    const readbackResultMatrix = device.createBuffer({
        size: resultSizeByte,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });

    const computePipeline = device.createComputePipeline({
        compute: {
            module: device.createShaderModule({
                code: tiled ? matMulShaderTiled : matMulShader
            }),
            entryPoint: 'main'
        }
    });

    const startTime = performance.now();
    const bindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: {
                    buffer: firstMatrixBuffer
                }
            },
            {
                binding: 1,
                resource: {
                    buffer: secondMatrixBuffer
                }
            },
            {
                binding: 2,
                resource: {
                    buffer: resultMatrixBuffer
                }
            }
        ]
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatch(Math.ceil(secondMatrix[1] / 16), Math.ceil(firstMatrix[0] / 16));
    passEncoder.endPass();

    commandEncoder.copyBufferToBuffer(resultMatrixBuffer, 0, readbackResultMatrix, 0, resultSizeByte);

    device.queue.submit([commandEncoder.finish()]);

    await readbackResultMatrix.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readbackResultMatrix.getMappedRange());
    const elapsedTime = performance.now() - startTime;

    // set size for completeness
    result[0] = firstMatrix[0];
    result[1] = secondMatrix[1];

    return {
        result,
        elapsedTime
    };
}
