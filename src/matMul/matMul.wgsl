
//@override
//let workGroupSize: u32 = 16u;
// see GPUProgrammableStage.constants

struct Matrix {
    numRows: f32;
    numCols: f32;
    data: array<f32>;
};

@group(0)
@binding(0)
var<storage, read> firstMatrix: Matrix;

@group(0)
@binding(1)
var<storage, read> secondMatrix: Matrix;

@group(0)
@binding(2)
var<storage, write> resultMatrix: Matrix;

@stage(compute)
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
    if (globalId.x >= u32(secondMatrix.numCols) || globalId.y >= u32(firstMatrix.numRows)) {
        return;
    }

    var result: f32 = 0.f;

    // y: row
    // x: col
    let resultCell = vec2<u32>(globalId.x, globalId.y);
    for (var i: u32 = 0u; i < u32(firstMatrix.numCols); i = i + 1u) {
        let a = i + resultCell.y * u32(firstMatrix.numCols);
        let b = i * u32(secondMatrix.numCols) + resultCell.x;
        result = result + firstMatrix.data[a] * secondMatrix.data[b];
    }

    let index = resultCell.x + resultCell.y * u32(secondMatrix.numCols);
    resultMatrix.data[index] = result;
}