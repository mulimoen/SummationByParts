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

    const program = ctx.createProgram();
    const vsShader = ctx.createShader(ctx.VERTEX_SHADER);
    ctx.shaderSource(vsShader, LINE_VERTEX_SHADER);
    const fsShader = ctx.createShader(ctx.FRAGMENT_SHADER);
    ctx.shaderSource(fsShader, LINE_FRAGMENT_SHADER);

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

    const uBbox = ctx.getUniformLocation(program, "uBbox");

    return {program: program, uniforms: {"uBbox": uBbox}};
};

(async function run() {
    const canvas = document.getElementById("glCanvas");

    const gl = canvas.getContext("webgl");
    if (gl === null) {
        console.error("Unable to initialise WebGL");
        return;
    }

    console.info("Successfully opened webgl canvas");

    const width = 23;
    const height = 21;

    const x = new Float32Array(width * height);
    const y = new Float32Array(width * height);
    for (let j = 0; j < height; j += 1) {
        for (let i = 0; i < width; i += 1) {
            const n = width*j + i;
            x[n] = i / (width - 1.0);
            y[n] = j / (height - 1.0);
        }
    }

    const bbox = [+Number(Infinity), -Number(Infinity), +Number(Infinity), -Number(Infinity)];
    for (let i = 0; i < width*height; i += 1) {
        bbox[0] = Math.min(bbox[0], x[i]);
        bbox[1] = Math.max(bbox[1], x[i]);
        bbox[2] = Math.min(bbox[2], y[i]);
        bbox[3] = Math.max(bbox[3], y[i]);
    }

    // A nice pink to show missing values
    gl.clearColor(1.0, 0.753, 0.796, 1.0);
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.disable(gl.CULL_FACE);

    const lineProg = lineProgram(gl);
    gl.useProgram(lineProg.program);


    // Transfer x, y to cpu, prepare fBuffer
    const xBuffer = gl.createBuffer();
    const yBuffer = gl.createBuffer();
    {
        let loc = gl.getAttribLocation(lineProg.program, "aX");
        gl.bindBuffer(gl.ARRAY_BUFFER, xBuffer);
        gl.vertexAttribPointer(loc, 1, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(loc);
        gl.bufferData(gl.ARRAY_BUFFER, x, gl.STATIC_DRAW);

        loc = gl.getAttribLocation(lineProg.program, "aY");
        gl.bindBuffer(gl.ARRAY_BUFFER, yBuffer);
        gl.vertexAttribPointer(loc, 1, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(loc);
        gl.bufferData(gl.ARRAY_BUFFER, y, gl.STATIC_DRAW);
    }

    // Create triangles covering the domain
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
    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, lineIdxBuffer, gl.STATIC_DRAW);

    gl.lineWidth(1.0);

    gl.uniform4f(lineProg.uniforms.uBbox, bbox[0], 1.0/(bbox[1] - bbox[0]), bbox[2], 1.0/(bbox[3] - bbox[2]));


    function drawMe() {
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        gl.drawElements(gl.LINE_STRIP, 2*width*height - 1, gl.UNSIGNED_SHORT, 0);

        window.requestAnimationFrame(drawMe);
    }

    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    }


    resizeCanvas();
    window.addEventListener("resize", resizeCanvas, false);
    window.requestAnimationFrame(drawMe);

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

    const play_button = document.getElementById("toggle-playing");
    const reset_button = document.getElementById("reset");
}());
