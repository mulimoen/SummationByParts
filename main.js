main();

function main() {
    const canvas = document.querySelector("#glCanvas");

    const gl = canvas.getContext("webgl");

    if (gl === null) {
        alert("Unable to initialise WebGL");
        return;
    }

    const vsSource = `
        attribute vec2 aVertexPosition;
        varying lowp vec2 vVertexPosition;

        void main() {
            vVertexPosition = aVertexPosition;
            gl_Position = vec4(aVertexPosition.x, aVertexPosition.y, 0.0, 1.0);
        }
    `;

    const fsSource = `
        varying lowp vec2 vVertexPosition;

        void main() {
            lowp float x = (vVertexPosition.x + 1.0)/2.0;
            lowp float y = (vVertexPosition.y + 1.0)/2.0;
            gl_FragColor = vec4(x, y, mod(x + y, 0.5), 1.0);
            //gl_FragColor = vec4(1.0, 0.0, 1.0, 1.0);
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

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clearDepth(1.0);
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

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

    gl.useProgram(programInfo.program);

    const offset = 0;
    const vertexCount = 4;
    gl.drawArrays(gl.TRIANGLE_STRIP, offset, vertexCount);
}
