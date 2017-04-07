"use strict";

class ReactionDiffusion {
	constructor(cvs) {
		this.canvas = cvs;
		this.gl = cvs.getContext("webgl2") || cvs.getContext("webgl-experimental");
		if (!this.gl) {
			alert("This browser does not support WebGL");
			return;
		}

		this.width = cvs.width;
		this.height = cvs.height;
		this.numOscillators = 88;
		this.oscillators = [];
		this.oscillators.fill(0, 0, 60);
		this.a = Math.pow(2, 1/12.0);
		//convenience
		let gl = this.gl;

		this.gl.clearColor(0.0, 0.0, 0.0, 1.0);
		this.gl.frontFace(gl.CW);

		const quad = [
			-1, 1,
			1, 1,
			-1, -1,
			1, 1,
			1, -1,
			-1, -1
		];
		let quadBuf = new ArrayBuffer(4*quad.length);
		let fView = new Float32Array(quadBuf);
		fView.set(quad);

		this.vBuf = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, this.vBuf);
		gl.bufferData(gl.ARRAY_BUFFER, quadBuf, gl.STATIC_DRAW);

		const line = new Float32Array([0, 0, 0, 1]);
		this.lineBuf = gl.createBuffer();
		gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
		gl.bufferData(gl.ARRAY_BUFFER, line, gl.STATIC_DRAW);

		let vertShaderSource = "" +
			"precision highp float;" +
			"precision highp int;" +
			"attribute vec2 vPos;" +
			"void main() {" +
			"   gl_Position = vec4(vPos, 0.0, 1.0);" +
			"}";

		let diffusionShaderSource = "" +
			"precision highp float;" +
			"precision highp int;" +
			"" +
			"uniform float da;" +
			"uniform float db;" +
			"uniform float f;" +
			"uniform float k;" +
			"uniform float dt;" +
			"uniform vec2 dim;" +
			"uniform sampler2D prevFrame;" +
			"" +
			"void main() {" +
			"   vec2 d = 1.0/dim;" +
			"   vec2 uv = gl_FragCoord.xy/dim;" +
			"" +
			"   vec2 tl = 0.05 * texture2D(prevFrame, uv+vec2(-d.x, d.y)).rb;" +
			"   vec2 tc = 0.2 * texture2D(prevFrame, uv+vec2(0, d.y)).rb;" +
			"   vec2 tr = 0.05 * texture2D(prevFrame, uv+vec2(d.x, d.y)).rb;" +
			"   vec2 cl = 0.2 * texture2D(prevFrame, uv+vec2(-d.x, 0)).rb;" +
			"   vec2 cc = -1.0 * texture2D(prevFrame, uv+vec2(0, 0)).rb;" +
			"   vec2 cr = 0.2 * texture2D(prevFrame, uv+vec2(d.x, 0)).rb;" +
			"   vec2 bl = 0.05 * texture2D(prevFrame, uv+vec2(-d.x, -d.y)).rb;" +
			"   vec2 bc = 0.2 * texture2D(prevFrame, uv+vec2(0, -d.y)).rb;" +
			"   vec2 br = 0.05 * texture2D(prevFrame, uv+vec2(d.x,- d.y)).rb;" +
			"" +
			"   vec2 sum = (tl + tc + tr + cl + cc + cr + bl + bc + br);" +
			"   vec2 thisCell = texture2D(prevFrame, uv).rb;" +
			"   float powG = thisCell.g*thisCell.g;" +
			"   float nextA = thisCell.r + (da*sum.r - thisCell.r*powG + f*(1.0-thisCell.r))*dt;" +
			"   float nextB = thisCell.g + (db*sum.g + thisCell.r*powG - (k+f)*thisCell.g)*dt;" +
			"   gl_FragColor = vec4(nextA, 0.0, nextB, 1.0);" +
			"}";

		let rtsFragShaderSource = "" +
			"precision highp float;" +
			"precision highp int;" +
			"uniform vec2 dim;" +
			"uniform sampler2D frame;" +
			"void main() {" +
			"   vec2 uv = gl_FragCoord.xy/dim;" +
			"   vec2 pixel = texture2D(frame, uv).rb;" +
			"   gl_FragColor = vec4(vec3(pixel.g), 1.0);" +
			"}";

		let lineVertShaderSource = "" +
			"precision highp float;" +
			"precision highp int;" +
			"attribute vec2 vPos;" +
			"uniform mat2 transform;" +
			"void main() {" +
			"   gl_Position = vec4(transform*vPos, 0.0, 1.0);" +
			"}";

		let lineFragShaderSource = "" +
			"precision highp float;" +
			"precision highp int;" +
			"" +
			"void main() {" +
			"   gl_FragColor = vec4(0.0, 1.0, 0.0, 1.0);" +
			"}";

		this.rtsProgram = this.CreateShaderProgram(vertShaderSource, rtsFragShaderSource);
		this.EnumerateUniforms(this.rtsProgram, ["dim", "frame"]);

		this.diffusionProgram = this.CreateShaderProgram(vertShaderSource, diffusionShaderSource);
		this.EnumerateUniforms(this.diffusionProgram, ["da", "db", "f", "k", "dt", "dim", "prevFrame"]);

		this.lineProgram = this.CreateShaderProgram(lineVertShaderSource, lineFragShaderSource);
		this.EnumerateUniforms(this.lineProgram, ["transform"]);

		//Hook up vertex attribute (only need one since we're just drawing a quad)
		this.attribLoc = 0;
		gl.bindAttribLocation(this.rtsProgram, this.attribLoc, "vPos");
		gl.bindAttribLocation(this.diffusionProgram, this.attribLoc, "vPos");
		gl.bindAttribLocation(this.lineProgram, this.attribLoc, "vPos");

		//Create a local framebuffer to swap out color buffers
		this.frameBuffer = gl.createFramebuffer();
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.frameBuffer);
		this.frameBuffer.depth = gl.createRenderbuffer();
		gl.bindRenderbuffer(gl.RENDERBUFFER, this.frameBuffer.depth);
		gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, this.width, this.height);
		gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, this.frameBuffer.depth);

		this.buffers = [];
		for(let i = 0; i < 2; i++) {
			let buf = gl.createTexture();
			gl.activeTexture(gl.TEXTURE0 + i);
			gl.bindTexture(gl.TEXTURE_2D, buf);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
			gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
			gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
			gl.generateMipmap(gl.TEXTURE_2D);
			this.buffers.push(buf);
		}
		this.prevBuffer = 1;

		let dataArray = [];
		let centerX = this.width/2;
		let centerY = this.height/2;
		for (let y = 0; y < this.height; y++) {
			for (let x = 0; x < this.width; x++) {
				//let f = (Math.random() > 0.99)*255;

				let b = 0;
				let diffX = x - centerX;
				let diffY = y - centerY;
				let dist = diffX*diffX + diffY*diffY;
				if (dist < 100)
					b = 255;

				dataArray.push(255, 0, b, 255);
			}
		}
		let data = new Uint8Array(dataArray);

		gl.activeTexture(gl.TEXTURE0 + this.prevBuffer);
		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);

		gl.viewport(0, 0, cvs.width, cvs.height);

		//audio resources
		this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
		this.analyser = this.audioCtx.createAnalyser();
		this.analyser.connect(this.audioCtx.destination);

		for (let i = 0; i < this.numOscillators; i++) {
			let o = this.audioCtx.createOscillator();
			o.frequency.value = 27.5 * Math.pow(this.a, i);
			let g = this.audioCtx.createGain();
			g.gain.value = 0.0;
			o.connect(g);
			o.start();
			g.connect(this.audioCtx.destination);
			this.oscillators[i] = g;
		}

		this.pixels = new Uint8Array(this.width*this.height*4);
		this.audioBuffer = this.audioCtx.createBuffer(1, this.audioCtx.sampleRate/60, this.audioCtx.sampleRate);
		this.radarAngle = Math.PI/2;
	}
	
	CreateShaderProgram(vertSource, fragSource) {
		let gl = this.gl;
		let vertShader = gl.createShader(gl.VERTEX_SHADER);
		let fragShader = gl.createShader(gl.FRAGMENT_SHADER);
		gl.shaderSource(vertShader, vertSource);
		gl.shaderSource(fragShader, fragSource);
		gl.compileShader(vertShader);
		gl.compileShader(fragShader);

		if (gl.getShaderParameter(vertShader, gl.COMPILE_STATUS)) {
			console.log("Vertex shader compile succeeded.")
		} else {
			console.log("Vertex shader compile failed.");
			console.log(gl.getShaderInfoLog(vertShader));
		}

		if (gl.getShaderParameter(fragShader, gl.COMPILE_STATUS)) {
			console.log("Fragment shader compile succeeded.")
		} else {
			console.log("Fragment shader compile failed.");
			console.log(gl.getShaderInfoLog(fragShader));
		}

		//Create the shader program
		let program = gl.createProgram();
		gl.attachShader(program, vertShader);
		gl.attachShader(program, fragShader);
		gl.linkProgram(program);

		if (gl.getProgramParameter(program, gl.LINK_STATUS)) {
			console.log("Shader program link succeeded.");
		} else {
			console.log("Shader program link failed.");
			console.log(gl.getProgramInfoLog(program));
		}

		return program;
	}

	EnumerateUniforms(program, uniforms) {
		uniforms.forEach((e, i, a) => {
			program[e] = this.gl.getUniformLocation(program, e);
		});
	}

	Run(time) {
		let da = 0.4,
			db = 0.20,
			f = 0.029,
			k = 0.057;

		let gl = this.gl;
		if (!this.start) this.start = time;
		let dt = 1.5;
		this.start = time;
		console.log(dt);

		let currentBuffer = (this.prevBuffer + 1) % 2;

		//Diffusion pass
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.frameBuffer);
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.buffers[currentBuffer], 0);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.vBuf);
		gl.useProgram(this.diffusionProgram);
		gl.vertexAttribPointer(this.attribLoc, 2, gl.FLOAT, gl.FALSE, 0, 0);
		gl.enableVertexAttribArray(this.attribLoc);
		gl.uniform1f(this.diffusionProgram.da, this.da);
		gl.uniform1f(this.diffusionProgram.db, this.db);
		gl.uniform1f(this.diffusionProgram.f, this.f);
		gl.uniform1f(this.diffusionProgram.k, this.k);
		gl.uniform1f(this.diffusionProgram.dt, this.dt);
		gl.uniform2f(this.diffusionProgram.dim, this.width, this.height);
		gl.uniform1i(this.diffusionProgram.prevFrame, this.prevBuffer);
		gl.activeTexture(gl.TEXTURE0 + this.prevBuffer);
		gl.viewport(0, 0, this.width, this.height);
		gl.drawArrays(gl.TRIANGLES, 0, 6);

		//Render to screen
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.clear(gl.COLOR_BUFFER_BIT);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.vBuf);
		gl.useProgram(this.rtsProgram);
		gl.vertexAttribPointer(this.attribLoc, 2, gl.FLOAT, gl.FALSE, 0, 0);
		gl.enableVertexAttribArray(this.attribLoc);
		gl.uniform2f(this.rtsProgram.dim, this.canvas.width, this.canvas.height);
		gl.uniform1i(this.rtsProgram.frame, this.buffers[currentBuffer]);
		gl.activeTexture(gl.TEXTURE0 + currentBuffer);
		gl.viewport(0, 0, this.canvas.width, this.canvas.height);
		gl.drawArrays(gl.TRIANGLES, 0, 6);

		/*
		let maxMipLevel = Math.floor(Math.log2(this.width));
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.activeTexture(gl.TEXTURE0 + 2);
		gl.copyTexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 0, 0, this.width, this.height, 0);
		gl.generateMipmap(gl.TEXTURE_2D);
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.frameBuffer);
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.copyBuffer, 0);
		gl.activeTexture(gl.TEXTURE0 + 3);
		gl.copyTexImage2D(gl.TEXTURE_2D, maxMipLevel, gl.RGBA, 0, 0, 1, 1, 0);
		*/

		let r = Math.min(Math.abs(1/Math.cos(this.radarAngle)), Math.abs(1/Math.sin(this.radarAngle)));
		let centerX = this.width/2;
		let centerY = this.height/2;
		let cornerRadius = Math.sqrt(2) * centerX;
		let c = r*Math.cos(this.radarAngle);
		let s = r*Math.sin(this.radarAngle);
		let mat = new Float32Array([c, -s, s, c]);
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
		gl.useProgram(this.lineProgram);
		gl.vertexAttribPointer(this.attribLoc, 2, gl.FLOAT, gl.FALSE, 0, 0);
		gl.enableVertexAttribArray(this.attribLoc);
		gl.uniformMatrix2fv(this.lineProgram.transform, false, mat);
		gl.drawArrays(gl.LINES, 0, 2);

		let oscillators = [];
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.frameBuffer);
		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.buffers[currentBuffer], 0);
		gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.UNSIGNED_BYTE, this.pixels);
		let dr = r / this.numOscillators;
		for (let o = 0; o < this.numOscillators; o++)
		{
			this.oscillators[o].gain.value = 0.0;
			let ratio = o / (this.numOscillators-1.0);
			let x = Math.floor(ratio*s*centerX) + centerX;
			let y = Math.floor(ratio*c*centerY) + centerY;
			if (x < 0 || x >= this.width || y < 0 || y >= this.height)
				continue;

			let idx = (x + y * this.width)*4;
			let pixel = [this.pixels[idx], this.pixels[idx+1], this.pixels[idx+2], this.pixels[idx+3]];
			if (pixel[2] > 0) {
				this.oscillators[o].gain.value = 0.05;
			}
		}

		/*let srcNode = this.audioCtx.createBufferSource();
		srcNode.buffer = this.audioBuffer;
		srcNode.connect(this.audioCtx.destination);
		srcNode.start();*/

		this.radarAngle += 0.016 * Math.PI;
		this.prevBuffer = currentBuffer;
		window.requestAnimationFrame(this.Run.bind(this));
	}
}

//Throwing this in global because module.exports doesn't work for some reason
global.ReactionDiffusion = ReactionDiffusion;