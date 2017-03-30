"use strict";

require("./diffusion.js");

document.addEventListener("DOMContentLoaded", function() {
	const cvs = document.querySelector("canvas");
	cvs.width = 2048;
	cvs.height = 1024;

	const rd = new ReactionDiffusion(cvs);
	window.requestAnimationFrame(rd.Run.bind(rd));

});


