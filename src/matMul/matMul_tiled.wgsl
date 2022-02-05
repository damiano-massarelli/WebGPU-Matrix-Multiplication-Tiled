let TILE_SIZE: u32 = 16u;

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

var<workgroup> firstTile : array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> secondTile : array<array<f32, TILE_SIZE>, TILE_SIZE>;

@stage(compute)
@workgroup_size(TILE_SIZE, TILE_SIZE)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>,
  @builtin(workgroup_id) blockId : vec3<u32>,
  @builtin(local_invocation_id) threadId : vec3<u32>) {
    var result: f32 = 0.f;



    for (var tile: u32 = 0u; tile < u32(firstMatrix.numCols); tile = tile + TILE_SIZE) {
        let firstRow = globalId.y;
        let firstCol = threadId.x + tile;

        let secondRow = threadId.y + tile;
        let secondCol = globalId.x;

        firstTile[threadId.y][threadId.x] = 0.0;
        if (firstRow < u32(firstMatrix.numRows) && firstCol < u32(firstMatrix.numCols)) {
            firstTile[threadId.y][threadId.x] = firstMatrix.data[firstRow * u32(firstMatrix.numCols) + firstCol];
        }

        secondTile[threadId.y][threadId.x] = 0.0;
        if (secondRow < u32(secondMatrix.numRows) && secondCol < u32(secondMatrix.numCols)) {
            secondTile[threadId.y][threadId.x] = secondMatrix.data[secondRow * u32(secondMatrix.numCols) + secondCol];
        }

        workgroupBarrier();

        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            result = result + firstTile[threadId.y][i] * secondTile[i][threadId.x];
        }

        workgroupBarrier();
    }

    if (globalId.x < u32(secondMatrix.numCols) && globalId.y <  u32(firstMatrix.numRows)) {
        let index = globalId.x + globalId.y * u32(secondMatrix.numCols);
        resultMatrix.data[index] = result;
    }
}