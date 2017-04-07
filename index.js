"use strict";

require("./diffusion.js");

//flp2 impl from Hacker's Delight
function NextLowestPoT(x)
{
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return x - (x >> 1);
}

document.addEventListener("DOMContentLoaded", function() {
	const cvs = document.querySelector("canvas");
	let bestSize = NextLowestPoT(Math.min(window.innerWidth/1.1, window.innerHeight/1.1));
	cvs.width = bestSize;
	cvs.height = bestSize;
	cvs.marginLeft = window.innerWidth/4;

	const rd = new ReactionDiffusion(cvs);
	window.requestAnimationFrame(rd.Run.bind(rd));
	let daInput = document.querySelector("#da");
	let dbInput = document.querySelector("#db");
	let fInput = document.querySelector("#f");
	let kInput = document.querySelector("#k");
	let dtInput = document.querySelector("#dt");

	document.querySelector("#enterButton").addEventListener("click", function(){
		let da = daInput.value != "" ? parseFloat(daInput.value) : null;
		let db = dbInput.value != "" ? parseFloat(dbInput.value) : null;
		let f = fInput.value != "" ? parseFloat(fInput.value) : null;
		let k = kInput.value != "" ? parseFloat(kInput.value) : null;
		let dt = dtInput.value != "" ? parseFloat(dtInput.value) : null;

		rd.UpdateParameters(da, db, f, k, dt);
		rd.ResetBuffers();
	});
});


