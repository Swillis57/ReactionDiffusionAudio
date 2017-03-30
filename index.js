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
	let bestSize = NextLowestPoT(Math.min(window.innerWidth, window.innerHeight));
	cvs.width = bestSize;
	cvs.height = bestSize;
	cvs.marginLeft = window.innerWidth/4;

	const rd = new ReactionDiffusion(cvs);
	window.requestAnimationFrame(rd.Run.bind(rd));

});


