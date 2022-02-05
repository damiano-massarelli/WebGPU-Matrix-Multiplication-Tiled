# WebGPU Matrix Multiplication

Matrix multiplication with WebGPU using the naive and [tiled](https://penny-xu.github.io/blog/tiled-matrix-multiplication) algorithms.

## How to run
```
npm install
npm run prod
npx serve dist
```

## Notices
- At the moment WebGPU is only available on browsers nighly builds and canary releases. This code has been tested on Chrome Canary 100.0.4871.0.
- WebGPU is still under development and its API / shading language may change.