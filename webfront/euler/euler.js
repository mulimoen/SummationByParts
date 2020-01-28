import { EulerUniverse, default as init, set_panic_hook as setPanicHook } from "../sbp.js";

/**
 * Initialises and runs the Euler solver,
 * plotting the solution to a canvas using webgl
 */
(async function run() {
    const wasm = await init("../sbp_bg.wasm");
    setPanicHook();
    const DIAMOND = false;
    const UPWIND = true;

    const canvas = document.getElementById("glCanvas");

    const gl = canvas.getContext("webgl");
    if (gl === null) {
        console.error("Unable to initialise WebGL");
        return;
    }

    const vsSource = String.raw`
        #version 100
        attribute mediump float aX;
        attribute mediump float aY;
        attribute mediump float aField;

        uniform vec4 uBbox;

        varying mediump float vField;

        void main() {
            vField = aField;
            mediump float x = (aX - uBbox.x)*uBbox.y;
            mediump float y = (aY - uBbox.z)*uBbox.w;
            gl_Position = vec4(2.0*x - 1.0, 2.0*y - 1.0, 1.0, 1.0);
        }
    `;

    const fsSource = String.raw`
        #version 100
        varying mediump float vField;

        uniform int uChosenField;

        void main() {
            mediump float r = 0.0;
            mediump float g = 0.0;
            mediump float b = 0.0;

            if (uChosenField == 0) {
                r = vField + 0.5;
            } else if (uChosenField == 1) {
                g = 2.5*(vField - 1.0) + 0.5;
            } else {
                b = vField + 0.5;
            }
            gl_FragColor = vec4(r, g, b, 1.0);
        }
    `;

    const vsShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vsShader, vsSource);
    gl.compileShader(vsShader);
    if (!gl.getShaderParameter(vsShader, gl.COMPILE_STATUS)) {
        console.error(`Could not compile shader: ${gl.getShaderInfoLog(vsShader)}`);
        return;
    }

    const fsShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fsShader, fsSource);
    gl.compileShader(fsShader);
    if (!gl.getShaderParameter(fsShader, gl.COMPILE_STATUS)) {
        console.error(`Could not compile shader: ${gl.getShaderInfoLog(fsShader)}`);
        return;
    }

    const shaderProgram = gl.createProgram();
    gl.attachShader(shaderProgram, vsShader);
    gl.attachShader(shaderProgram, fsShader);
    gl.linkProgram(shaderProgram);
    gl.validateProgram(shaderProgram);
    if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
        console.error(`Unable to link shader program: ${gl.getProgramInfoLog(shaderProgram)}`);
        return;
    }
    gl.useProgram(shaderProgram);

    // A nice pink to show missing values
    gl.clearColor(1.0, 0.753, 0.796, 1.0);
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.disable(gl.CULL_FACE);

    console.info("Successfully set OpenGL state");


    const width = 40;
    const height = 50;


    const x = new Float32Array(width * height);
    const y = new Float32Array(width * height);
    for (let j = 0; j < height; j += 1) {
        for (let i = 0; i < width; i += 1) {
            const n = width*j + i;
            x[n] = 20.0*(i / (width - 1.0) - 0.5);
            y[n] = 20.0*(j / (height - 1.0) - 0.5);


            if (DIAMOND) {
                const xx = x[n];
                const yy = y[n];
                x[n] = xx - yy;
                y[n] = xx + yy;
            }
        }
    }

    const universe = new EulerUniverse(height, width, x, y);


    // Transfer x, y to cpu, prepare fBuffer
    const xBuffer = gl.createBuffer();
    const yBuffer = gl.createBuffer();
    const fBuffer = gl.createBuffer();
    {
        const numcomp = 1;
        const type = gl.FLOAT;
        const normalise = false;
        const stride = 0;
        const offset = 0;

        let loc = gl.getAttribLocation(shaderProgram, "aX");
        gl.bindBuffer(gl.ARRAY_BUFFER, xBuffer);
        gl.vertexAttribPointer(loc, numcomp, type, normalise, stride, offset);
        gl.enableVertexAttribArray(loc);
        gl.bufferData(gl.ARRAY_BUFFER, x, gl.STATIC_DRAW);

        loc = gl.getAttribLocation(shaderProgram, "aY");
        gl.bindBuffer(gl.ARRAY_BUFFER, yBuffer);
        gl.vertexAttribPointer(loc, numcomp, type, normalise, stride, offset);
        gl.enableVertexAttribArray(loc);
        gl.bufferData(gl.ARRAY_BUFFER, y, gl.STATIC_DRAW);

        loc = gl.getAttribLocation(shaderProgram, "aField");
        gl.bindBuffer(gl.ARRAY_BUFFER, fBuffer);
        gl.vertexAttribPointer(loc, numcomp, type, normalise, stride, offset);
        gl.enableVertexAttribArray(loc);
    }

    // Create triangles covering the domain
    const positions = new Int16Array((width-1)*(height-1)*2*3);
    for (let j = 0; j < height - 1; j += 1) {
        for (let i = 0; i < width - 1; i += 1) {
            const n = 2*3*((width-1)*j + i);
            positions[n+0] = width*j + i;
            positions[n+1] = width*j + i+1;
            positions[n+2] = width*(j+1) + i;
            positions[n+3] = width*j + i+1;
            positions[n+4] = width*(j+1) + i;
            positions[n+5] = width*(j+1) + i+1;
        }
    }

    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const bbox = [+Number(Infinity), -Number(Infinity), +Number(Infinity), -Number(Infinity)];
    for (let i = 0; i < width*height; i += 1) {
        bbox[0] = Math.min(bbox[0], x[i]);
        bbox[1] = Math.max(bbox[1], x[i]);
        bbox[2] = Math.min(bbox[2], y[i]);
        bbox[3] = Math.max(bbox[3], y[i]);
    }

    {
        const loc = gl.getUniformLocation(shaderProgram, "uBbox");
        gl.uniform4f(loc, bbox[0], 1.0/(bbox[1] - bbox[0]), bbox[2], 1.0/(bbox[3] - bbox[2]));
    }

    const TIMEFACTOR = 10000.0;
    const MAX_DT = Math.min(1.0/width, 1.0/height)*0.2;

    let t = 0;
    let firstDraw = true;
    let warnTime = -1;
    const chosenField = {
        "uLocation": gl.getUniformLocation(shaderProgram, "uChosenField"),
        "value": 0,
        "cycle": function cycle() {
            if (this.value === 0) {
                this.value = 1;
            } else if (this.value === 1) {
                this.value = 2;
            } else {
                this.value = 0;
            }
            gl.uniform1i(this.uLocation, this.value);
        }
    };
    chosenField.cycle();

    universe.init(0, 0);

    /**
     * Integrates and draws the next iteration
     * @param {Time} timeOfDraw Time of drawing
     */
    function drawMe(timeOfDraw) {
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        let dt = 0.01;

        let fieldPtr;
        if (chosenField.value === 0) {
            fieldPtr = universe.get_rho_ptr();
        } else if (chosenField.value === 1) {
            fieldPtr = universe.get_rhou_ptr();
        } else if (chosenField.value == 2) {
            fieldPtr = universe.get_rhov_ptr();
        } else {
            fieldPtr = universe.get_e_ptr();
        };
        const field = new Float32Array(wasm.memory.buffer, fieldPtr, width*height);
        gl.bufferData(gl.ARRAY_BUFFER, field, gl.DYNAMIC_DRAW);
        // console.log(field.reduce((min, v) => v < min ? v : min));
        // console.log(field.reduce((max, v) => v > max ? v : max));

        {
            const offset = 0;
            const type = gl.UNSIGNED_SHORT;
            const vertexCount = positions.length;
            gl.drawElements(gl.TRIANGLES, vertexCount, type, offset);
        }

        if (UPWIND) {
            universe.advance_upwind(MAX_DT);
            universe.advance_upwind(MAX_DT);
        } else {
            universe.advance(MAX_DT);
            universe.advance(MAX_DT);
        }

        window.requestAnimationFrame(drawMe);
    }

    // https://stackoverflow.com/questions/4288253/html5-canvas-100-width-height-of-viewport#8486324
    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    }

    window.addEventListener("keyup", event => {
        if (event.key === "c") {
            chosenField.cycle();
        }
    }, {"passive": true});
    window.addEventListener("resize", resizeCanvas, false);
    window.addEventListener("click", event => {
        // Must adjust for bbox and transformations for x/y
        const mousex = event.clientX / window.innerWidth;
        const mousey = event.clientY / window.innerHeight;

        const normx = mousex;
        const normy = 1.0 - mousey;

        universe.init(
            (bbox[1] - bbox[0])*normx + bbox[0],
            (bbox[3] - bbox[2])*normy + bbox[2],
        );
    }, {"passive": true});

    resizeCanvas();
    window.requestAnimationFrame(drawMe);
}());
