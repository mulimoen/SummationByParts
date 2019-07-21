import { Universe, default as init } from "./webgl.js";

async function run() {
    let wasm = await init("./webgl_bg.wasm");

    const canvas = document.getElementById("glCanvas");

    const gl = canvas.getContext("webgl");
    const float_in_image = gl.getExtension("OES_texture_float");
    if (!float_in_image) {
        console.warn("Floats are not supported in images for your device, please warn the author about this incompatability");
    }

    if (gl === null) {
        alert("Unable to initialise WebGL");
        return;
    }

    const vsSource = `
        attribute vec2 aVertexPosition;
        varying lowp vec2 vVertexPosition;

        void main() {
            vVertexPosition = (aVertexPosition + 1.0)/2.0;
            gl_Position = vec4(aVertexPosition.x, aVertexPosition.y, 0.0, 1.0);
        }
    `;

    const fsSource = `
        varying lowp vec2 vVertexPosition;

        uniform sampler2D uSampler;

        void main() {
            mediump float c = texture2D(uSampler, vVertexPosition).a;
            gl_FragColor = vec4((c + 1.0)/2.0, 0.0, 0.0, 1.0);
        }
    `;

    const vsShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vsShader, vsSource);
    gl.compileShader(vsShader);
    if (!gl.getShaderParameter(vsShader, gl.COMPILE_STATUS)) {
        alert('Could not compile shader: ' + gl.getShaderInfoLog(vsShader));
        return;
    }

    const fsShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fsShader, fsSource);
    gl.compileShader(fsShader);
    if (!gl.getShaderParameter(fsShader, gl.COMPILE_STATUS)) {
        alert('Could not compile shader: ' + gl.getShaderInfoLog(fsShader));
        return;
    }

    const shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vsShader);
    gl.attachShader(shaderProgram, fsShader);
    gl.linkProgram(shaderProgram);

    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        alert('Unable to link shader program: ' + gl.getProgramInfoLog(shaderProgram))
        return;
    }

    const programInfo = {
        program: shaderProgram,
        attribLocations: {
            vertexPosition: gl.getAttribLocation(shaderProgram, 'aVertexPosition'),
        },
        uniformLocation: {
            sampler: gl.getUniformLocation(shaderProgram, 'uShader'),
        },
    };

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    const positions = [
        -1.0, 1.0,
        1.0, 1.0,
        -1.0, -1.0,
        1.0, -1.0,
    ];

    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);


    const width = 40;
    const height = 50;
    const universe = Universe.new(width, height);

    const t = performance.now();
    universe.set_initial(t/1000.0);


    const field = new Float32Array(wasm.memory.buffer,
            universe.get_ptr(),
            width*height);

    const texture = gl.createTexture();
    {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        const level = 0;
        const internalFormat = gl.ALPHA;
        const border = 0;
        const srcFormat = gl.ALPHA;
        const srcType = gl.FLOAT;
        gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, width, height, border, srcFormat, srcType, field);
    }

    // GL state functions, must be moved to loop if changing
    gl.clearColor(1.0, 0.753, 0.796, 1.0);
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.disable(gl.CULL_FACE);

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    gl.useProgram(programInfo.program);

    { // Binding vertices
        const numComponents = 2;
        const type = gl.FLOAT;
        const normalise = false;
        const stride = 0;
        const offset = 0;
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.vertexAttribPointer(programInfo.attribLocations.vertexPosition, numComponents, type, normalise, stride, offset);
        gl.enableVertexAttribArray(programInfo.attribLocations.vertexPositions);
    }

    function drawMe(t) {

        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        universe.set_initial(t/1000.0);

        const field = new Float32Array(wasm.memory.buffer,
                universe.get_ptr(),
                width*height);
        {
            const level = 0;
            const internalFormat = gl.ALPHA;
            const border = 0;
            const srcFormat = internalFormat;
            const srcType = gl.FLOAT;
            gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, width, height, border, srcFormat, srcType, field);
        }

        const offset = 0;
        const vertexCount = 4;
        gl.drawArrays(gl.TRIANGLE_STRIP, offset, vertexCount);

        window.requestAnimationFrame(drawMe);
    }

    // https://stackoverflow.com/questions/4288253/html5-canvas-100-width-height-of-viewport#8486324
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    }

    window.addEventListener('resize', resizeCanvas, false);
    resizeCanvas();
    window.requestAnimationFrame(drawMe);
}

run();
