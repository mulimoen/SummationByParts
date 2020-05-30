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
                    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
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
        this.xBuffer = xBuffer;
        this.yBuffer = yBuffer;
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

        ctx.useProgram(this.program.inner);
        ctx.uniform4f(this.program.uniforms.uBbox, bbox[0], 1.0/(bbox[1] - bbox[0]), bbox[2], 1.0/(bbox[3] - bbox[2]));
        this.width = width;
        this.height = height;
    }
    draw() {
        const ctx = this.ctx;
        ctx.useProgram(this.program.inner);
        {
            let loc;
            loc = ctx.getAttribLocation(this.program.inner, "aX");
            ctx.bindBuffer(ctx.ARRAY_BUFFER, this.xBuffer);
            ctx.vertexAttribPointer(loc, 1, ctx.FLOAT, false, 0, 0);
            ctx.enableVertexAttribArray(loc);

            loc = ctx.getAttribLocation(this.program.inner, "aY");
            ctx.bindBuffer(ctx.ARRAY_BUFFER, this.yBuffer);
            ctx.vertexAttribPointer(loc, 1, ctx.FLOAT, false, 0, 0);
            ctx.enableVertexAttribArray(loc);
        }
        ctx.bindBuffer(ctx.ELEMENT_ARRAY_BUFFER, this.indexBuffer);

        ctx.drawElements(ctx.LINE_STRIP, 2*this.width*this.height - 1, ctx.UNSIGNED_SHORT, 0);
    }
}

class FieldDrawer {
    constructor(ctx) {
        this.ctx = ctx;

        const VERT_SHADER = String.raw`
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

        const FRAG_SHADER = String.raw`
            #version 100
            varying mediump float vField;

            uniform mediump vec3 uColor;
            uniform mediump float uFieldMin;
            uniform mediump float uFieldMax;

            void main() {
                mediump float v = (vField - uFieldMin) / (uFieldMax - uFieldMin);
                gl_FragColor = vec4(mix(vec3(0.0), uColor, v), 1.0);
            }
        `;

        const program = compile_and_link(ctx, VERT_SHADER, FRAG_SHADER);
        this.fBuffer = ctx.createBuffer();

        const uniforms = {
            uBbox: ctx.getUniformLocation(program, "uBbox"),
            uColor: ctx.getUniformLocation(program, "uColor"),
            uFieldMin: ctx.getUniformLocation(program, "uFieldMin"),
            uFieldMax: ctx.getUniformLocation(program, "uFieldMax"),
        };

        this.program = {inner: program, uniforms: uniforms};
    }
    set_xy(width, height, xBuffer, yBuffer, bbox) {
        const ctx = this.ctx;
        this.xBuffer = xBuffer;
        this.yBuffer = yBuffer;
        ctx.useProgram(this.program.inner);
        ctx.uniform4f(this.program.uniforms.uBbox, bbox[0], 1.0/(bbox[1] - bbox[0]), bbox[2], 1.0/(bbox[3] - bbox[2]));
        this.setColor(0.0, 1.0, 0.0);

        const positions = new Uint16Array((width-1)*(height-1)*2*3);
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
        this.positions = positions;
        this.indexBuffer = ctx.createBuffer();
        ctx.bindBuffer(ctx.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
        ctx.bufferData(ctx.ELEMENT_ARRAY_BUFFER, this.positions, ctx.STATIC_DRAW);
    }
    setColor() {
        this.ctx.uniform3f(this.program.uniforms.uColor, 0.0, 1.0, 0.0);
    }
    draw(field) {
        const ctx = this.ctx;
        ctx.useProgram(this.program.inner);
        {
            let loc;
            loc = ctx.getAttribLocation(this.program.inner, "aX");
            ctx.bindBuffer(ctx.ARRAY_BUFFER, this.xBuffer);
            ctx.vertexAttribPointer(loc, 1, ctx.FLOAT, false, 0, 0);
            ctx.enableVertexAttribArray(loc);

            loc = ctx.getAttribLocation(this.program.inner, "aY");
            ctx.bindBuffer(ctx.ARRAY_BUFFER, this.yBuffer);
            ctx.vertexAttribPointer(loc, 1, ctx.FLOAT, false, 0, 0);
            ctx.enableVertexAttribArray(loc);

            loc = ctx.getAttribLocation(this.program.inner, "aField");
            ctx.bindBuffer(ctx.ARRAY_BUFFER, this.fBuffer);
            ctx.bufferData(ctx.ARRAY_BUFFER, field, ctx.DYNAMIC_DRAW);
            ctx.vertexAttribPointer(loc, 1, ctx.FLOAT, false, 0, 0);
            ctx.enableVertexAttribArray(loc);
        }

        ctx.bindBuffer(ctx.ELEMENT_ARRAY_BUFFER, this.indexBuffer);
        ctx.drawElements(ctx.TRIANGLES, this.positions.length, ctx.UNSIGNED_SHORT, 0);
    }
    minmax(min, max) {
        this.ctx.useProgram(this.program.inner);
        this.ctx.uniform1f(this.program.uniforms.uFieldMin, min);
        this.ctx.uniform1f(this.program.uniforms.uFieldMax, max);
    }
}

(async function run() {
    let is_setup = false;
    const eq_sel = document.getElementById("eq-set");
    display_eqset(eq_sel.value);

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
    const bbox = [+Number(Infinity), -Number(Infinity), +Number(Infinity), -Number(Infinity)];

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
    const fieldDrawer = new FieldDrawer(gl);
    let universe;

    let eq_set;

    function setup() {
        width = parseInt(document.getElementById("xN").value);
        height = parseInt(document.getElementById("yN").value);

        const x0 = parseFloat(document.getElementById("x0").value);
        const xn = parseFloat(document.getElementById("xn").value);
        const y0 = parseFloat(document.getElementById("y0").value);
        const yn = parseFloat(document.getElementById("yn").value);

        const diamond = document.getElementById("diamond").checked;

        x = new Float32Array(width * height);
        y = new Float32Array(width * height);
        const dx = (xn - x0) / (width - 1)
        const dy = (yn - y0) / (height - 1)
        for (let j = 0; j < height; j += 1) {
            for (let i = 0; i < width; i += 1) {
                const n = width*j + i;
                x[n] = x0 + dx*i;
                y[n] = y0 + dy*j;

                if (diamond) {
                    const xn = x[n];
                    const yn = y[n];

                    x[n] = xn - yn;
                    y[n] = xn + yn;
                }
            }
        }

        eq_set = eq_sel.value;
        switch (eq_set) {
            case "maxwell":
                universe = new MaxwellUniverse(height, width, x, y);
                universe.init(0.0, 0.0);
                break;
            case "euler":
                universe = new EulerUniverse(height, width, x, y);
                universe.init(0.0, 0.0);
                break;
            case "shallow":
                universe = new ShallowWaterUniverse(height, width);
                universe.init(0.0, 0.0);
                break;
            default:
                console.error(`Unknown case ${eq_set}`);
                return null;
        }

        bbox[0] = +Number(Infinity);
        bbox[1] = -Number(Infinity);
        bbox[2] = +Number(Infinity);
        bbox[3] = -Number(Infinity);
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
        fieldDrawer.set_xy(width, height, xBuffer, yBuffer, bbox);
    }

    function draw() {
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        canvas.width = w;
        canvas.height = h;
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);

        let fieldPtr;
        switch (eq_set) {
            case "maxwell":
                chosenField %= 3;
                switch (chosenField) {
                    case 0:
                        fieldPtr = universe.get_ex_ptr();
                        fieldDrawer.minmax(-1.0, 1.0);
                        break;
                    case 1:
                        fieldPtr = universe.get_hz_ptr();
                        fieldDrawer.minmax(-1.0, 1.0);
                        break;
                    case 2:
                        fieldPtr = universe.get_ey_ptr();
                        fieldDrawer.minmax(-1.0, 1.0);
                        break;
                }
                break;
            case "euler":
                chosenField %= 4;
                switch (chosenField) {
                    case 0:
                        fieldPtr = universe.get_rho_ptr();
                        fieldDrawer.minmax(0.95, 1.05);
                        break;
                    case 1:
                        fieldPtr = universe.get_rhou_ptr();
                        fieldDrawer.minmax(0.90, 1.10);
                        break;
                    case 2:
                        fieldPtr = universe.get_rhov_ptr();
                        fieldDrawer.minmax(-1.10, 1.10);
                        break;
                    case 3:
                        fieldPtr = universe.get_e_ptr();
                        fieldDrawer.minmax(7.0, 10.0);
                        break;
                }
                break;
            case "shallow":
                chosenField %= 3;
                switch (chosenField) {
                    case 0:
                        fieldPtr = universe.get_eta_ptr();
                        fieldDrawer.minmax(0.8, 1.1);
                        break;
                    case 1:
                        fieldPtr = universe.get_etau_ptr();
                        fieldDrawer.minmax(-0.5, 0.5);
                        break;
                    case 2:
                        fieldPtr = universe.get_etav_ptr();
                        fieldDrawer.minmax(-0.5, 0.5);
                        break;
                }
                break;
            default:
                console.error("Not implemented");
        }
        let field = new Float32Array(wasm.memory.buffer, fieldPtr, width*height);
        fieldDrawer.draw(field);
        lineDrawer.draw();
    }

    async function drawLoop() {
        draw();
        switch (eq_set) {
            case "maxwell":
                universe.advance(0.2*Math.min(1/width, 1/height));
                break;
            case "euler":
                universe.advance_upwind(0.2*Math.min(1/width, 1/height));
                break;
            case "shallow":
                universe.advance();
                break;
            default:
                console.error(`Not implemented: ${eq_set}`);
        }
        animation = window.requestAnimationFrame(drawLoop);
    }

    const menu = document.getElementById("menu");
    const menu_toggle = document.getElementById("toggle-menu");
    menu_toggle.addEventListener("click", () => {
        if (menu.style.visibility === "") {
            menu.style.visibility = "hidden";
        } else {
            menu.style.visibility = "";
        }
    }, {"passive": false});

    eq_sel.addEventListener("change", (e) => {
        display_eqset(eq_sel.value);
    }, {"passive": true});
    function display_eqset(value) {
        const euler_options = document.getElementById("euler-options");
        euler_options.style.display = "none";
        const maxwell_options = document.getElementById("maxwell-options");
        maxwell_options.style.display = "none";
        const shallow_options = document.getElementById("shallow-options");
        shallow_options.style.display = "none";
        function default_axes(x0, xn, y0, yn) {
            document.getElementById("x0").value = x0;
            document.getElementById("xn").value = xn;
            document.getElementById("y0").value = y0;
            document.getElementById("yn").value = yn;
        }
        if (value === "euler") {
            euler_options.style.display = "block";
            default_axes(-5.0, 5.0, -5.0, 5.0);
        } else if (value === "maxwell") {
            maxwell_options.style.display = "block";
            default_axes(-1.0, 1.0, -1.0, 1.0);
        } else if (value === "shallow") {
            shallow_options.style.display = "block";
            default_axes(-1.0, 1.0, -1.0, 1.0);
        } else {
            console.error(`Unknown value ${value}`);
        }
        is_setup = false;
    }

    let animation = null;
    const play_button = document.getElementById("toggle-playing");
    play_button.addEventListener("click", (_e) => {
        if (!animation) {
            if (!is_setup) {
                setup();
                is_setup = true;
            }
            animation = window.requestAnimationFrame(drawLoop);
        } else {
            if (is_setup) {
                window.cancelAnimationFrame(animation);
                animation = null;
            }
        }
    }, {"passive": true});
    const reset_button = document.getElementById("reset");
    reset_button.addEventListener("click", (_e) => {
        if (is_setup) {
            window.cancelAnimationFrame(animation);
            animation = null;
        }
        setup();
        is_setup = true;
        draw();
    }, {"passive": true});
    let chosenField = 0;
    const chosenFieldButton = document.getElementById("fieldButton");
    chosenFieldButton.addEventListener("click", (_e) => {
        chosenField += 1;
        if (is_setup) {
            draw();
        }
    }, {"passive": true});
    canvas.addEventListener("click", (e) => {
        if (is_setup) {
            const mousex = (e.clientX / canvas.clientWidth)*(bbox[1] - bbox[0]) + bbox[0];
            const mousey = (1.0 - e.clientY / canvas.clientHeight)*(bbox[3] - bbox[2]) + bbox[2];
            universe.init(mousex, mousey);
            draw();
        }
    }, {"passive": false});
}());
