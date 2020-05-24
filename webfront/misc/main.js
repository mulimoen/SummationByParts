import { default as init, set_panic_hook as setPanicHook,
         MaxwellUniverse, EulerUniverse, ShallowWaterUniverse } from "./sbp_web.js";

function compile_and_link(ctx, vsource, fsource) {
    const program = ctx.createProgram();
    const vsShader = ctx.createShader(ctx.VERTEX_SHADER);
    ctx.shaderSource(vsShader, vsource);
    const fsShader = ctx.createShader(ctx.FRAGMENT_SHADER);
    ctx.shaderSource(fsShader, fsource);

    ctx.compileShader(vsShader);
    ctx.compileShader(fsShader);
    ctx.attachShader(program, vsShader);
    ctx.attachShader(program, fsShader);
    ctx.linkProgram(program);
    ctx.validateProgram(program);

    if (!ctx.getProgramParameter(program, ctx.LINK_STATUS)) {
        console.error(`Linking of lineProgram failed: ${ctx.getProgramInfoLog(program)}`);
        console.error(`vertex shader infolog: ${ctx.getShaderInfoLog(vsShader)}`);
        console.error(`fragment shader infolog: ${ctx.getShaderInfoLog(fsShader)}`);
        return null;
    }

    ctx.deleteShader(vsShader);
    ctx.deleteShader(fsShader);

    return program;
}

class LineDrawer {
    constructor(ctx) {
        function lineProgram(ctx) {
            const LINE_VERTEX_SHADER = String.raw`
                #version 100
                attribute mediump float aX;
                attribute mediump float aY;

                uniform vec4 uBbox;

                void main() {
                    mediump float x = (aX - uBbox.x)*uBbox.y;
                    mediump float y = (aY - uBbox.z)*uBbox.w;
                    gl_Position = vec4(2.0*x - 1.0, 2.0*y - 1.0, 1.0, 1.0);
                }

            `;
            const LINE_FRAGMENT_SHADER = String.raw`
                #version 100

                void main() {
                    gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);
                }
            `;
            const program = compile_and_link(ctx, LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER);

            const uBbox = ctx.getUniformLocation(program, "uBbox");

            return {inner: program, uniforms: {"uBbox": uBbox}};
        };
        this.program = lineProgram(ctx);
        this.ctx = ctx;
    }
    set_xy(width, height, xBuffer, yBuffer, bbox) {
        const ctx = this.ctx;
        ctx.useProgram(this.program.inner);
        {
            let loc = ctx.getAttribLocation(this.program.inner, "aX");
            ctx.bindBuffer(ctx.ARRAY_BUFFER, xBuffer);
            ctx.vertexAttribPointer(loc, 1, ctx.FLOAT, false, 0, 0);
            ctx.enableVertexAttribArray(loc);

            loc = ctx.getAttribLocation(this.program.inner, "aY");
            ctx.bindBuffer(ctx.ARRAY_BUFFER, yBuffer);
            ctx.vertexAttribPointer(loc, 1, ctx.FLOAT, false, 0, 0);
            ctx.enableVertexAttribArray(loc);
        }
        const lineIdxBuffer = new Uint16Array(2*width*height - 1);
        {
            let n = 0;
            for (let j = 0; j < height; j++) {
                for (let i = 0; i < width; i++) {
                    if (j % 2 == 0) {
                        lineIdxBuffer[n] = width*j + i;
                    } else {
                        lineIdxBuffer[n] = width*(j + 1) - i - 1;
                    }
                    n += 1;
                }
            }
            let m = lineIdxBuffer[n-1];
            let updown = "down";
            let lr;
            if (height % 2 === 0) {
                lr = "left";
            } else {
                lr = "right";
            }
            for (let i = 0; i < width; i++) {
                for (let j = 0; j < height-1; j++) {
                    if (updown === "up") {
                        m += width;
                    } else {
                        m -= width;
                    }
                    lineIdxBuffer[n] = m;
                    n += 1;
                }
                if (n === 2*width*height - 1) {
                    continue;
                }
                if (lr === "left") {
                    m += 1;
                } else {
                    m -= 1;
                }
                lineIdxBuffer[n] = m;
                n += 1;

                if (updown === "down") {
                    updown = "up";
                } else {
                    updown = "down";
                }
            }
        }
        this.indexBuffer = ctx.createBuffer();
        ctx.bindBuffer(ctx.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
        ctx.bufferData(ctx.ELEMENT_ARRAY_BUFFER, lineIdxBuffer, ctx.STATIC_DRAW);

        ctx.uniform4f(this.program.uniforms.uBbox, bbox[0], 1.0/(bbox[1] - bbox[0]), bbox[2], 1.0/(bbox[3] - bbox[2]));
        this.width = width;
        this.height = height;
    }
    draw() {
        const ctx = this.ctx;
        ctx.useProgram(this.program.inner);
        ctx.bindBuffer(ctx.ELEMENT_ARRAY_BUFFER, this.indexBuffer);

        ctx.drawElements(ctx.LINE_STRIP, 2*this.width*this.height - 1, ctx.UNSIGNED_SHORT, 0);
    }
}

(async function run() {
    const wasm = await init("./sbp_web_bg.wasm");
    setPanicHook();
    const canvas = document.getElementById("glCanvas");

    const gl = canvas.getContext("webgl");
    if (gl === null) {
        console.error("Unable to initialise WebGL");
        return;
    }

    console.info("Successfully opened webgl canvas");

    let x;
    let y;
    let width;
    let height;

    // A nice pink to show missing values
    gl.clearColor(1.0, 0.753, 0.796, 1.0);
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.disable(gl.CULL_FACE);
    gl.lineWidth(1.0);

    const xBuffer = gl.createBuffer();
    const yBuffer = gl.createBuffer();
    const lineDrawer = new LineDrawer(gl);

    function setup() {
        width = parseInt(document.getElementById("xN").value);
        height = parseInt(document.getElementById("yN").value);

        const x0 = parseFloat(document.getElementById("x0").value);
        const xn = parseFloat(document.getElementById("xn").value);
        const y0 = parseFloat(document.getElementById("y0").value);
        const yn = parseFloat(document.getElementById("yn").value);

        const diamond = document.getElementById("diamond").checked;
        console.log(diamond);

        x = new Float32Array(width * height);
        y = new Float32Array(width * height);
        const dx = (xn - x0) / (width - 1)
        const dy = (yn - y0) / (height - 1)
        for (let j = 0; j < height; j += 1) {
            for (let i = 0; i < width; i += 1) {
                const n = width*j + i;
                x[n] = dx*i;
                y[n] = dy*j;

                if (diamond) {
                    const xn = x[n];
                    const yn = y[n];

                    x[n] = xn - yn;
                    y[n] = xn + yn;
                }
            }
        }

        const bbox = [+Number(Infinity), -Number(Infinity), +Number(Infinity), -Number(Infinity)];
        for (let i = 0; i < width*height; i += 1) {
            bbox[0] = Math.min(bbox[0], x[i]);
            bbox[1] = Math.max(bbox[1], x[i]);
            bbox[2] = Math.min(bbox[2], y[i]);
            bbox[3] = Math.max(bbox[3], y[i]);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, xBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, x, gl.STATIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, yBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, y, gl.STATIC_DRAW);

        lineDrawer.set_xy(width, height, xBuffer, yBuffer, bbox);
    }

    function draw() {
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        lineDrawer.draw();
    }

    function drawMe() {
        draw();
        animation = window.requestAnimationFrame(drawMe);
    }

    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    }


    resizeCanvas();
    window.addEventListener("resize", resizeCanvas, false);

    // window.requestAnimationFrame(drawMe);

    const menu = document.getElementById("menu");
    const menu_toggle = document.getElementById("toggle-menu");
    menu_toggle.addEventListener("click", () => {
        if (menu.style.visibility === "") {
            menu.style.visibility = "hidden";
        } else {
            menu.style.visibility = "";
        }
    });

    const eq_sel = document.getElementById("eq-set");
    eq_sel.addEventListener("change", (e) => {
        console.log("equation changed, wants: ", e.target.value);
    });

    let animation = null;
    let is_setup = false;
    const play_button = document.getElementById("toggle-playing");
    play_button.addEventListener("click", (_e) => {
        console.log("play/pause pressed");
        if (!animation) {
            if (!is_setup) {
                setup();
            }
            animation = window.requestAnimationFrame(drawMe);
        } else {
            window.cancelAnimationFrame(animation);
            animation = null;
        }
    });
    const reset_button = document.getElementById("reset");
    reset_button.addEventListener("click", (_e) => {
        window.cancelAnimationFrame(animation);
        animation = null;
        is_setup = false;
        setup();
        draw();
    });
}());
