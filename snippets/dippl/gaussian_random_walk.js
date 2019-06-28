// Gaussian Random Walk
var drawLines = function(canvas, start, positions){
  if (positions.length == 0) { return []; }
  var next = positions[0];
  canvas.line(start[0], start[1], next[0], next[1], 4, 0.2);
  drawLines(canvas, next, positions.slice(1));
  return;
}

var last = function(xs){
  return xs[xs.length - 1];
}
///

var init = function(dim){
  return repeat(dim, function(){ return gaussian(200, 1) });
}

var transition = function(pos){
  return map(
    function(x){ return gaussian(x, 10); },
    pos
  );
};

var gaussianRandomWalk = function(n, dim) {
  var prevStates = (n==1) ? [init(dim)] : gaussianRandomWalk(n-1, dim);
  var newState = transition(last(prevStates));
  return prevStates.concat([newState]);
};

var positions = gaussianRandomWalk(100, 2);

// Draw model output
var canvas = Draw(400, 400, true)
drawLines(canvas, positions[0], positions.slice(1))


// Semi-Markov Random Walk
var init = function(dim){
  return repeat(dim, function(){ return gaussian(200, 1) });
}

var transition = function(lastPos, secondLastPos){
  return map2(
    function(lastX, secondLastX){
      var momentum = (lastX - secondLastX) * .7;
      return gaussian(lastX + momentum, 3);
    },
	lastPos,
    secondLastPos
  );
};

var semiMarkovWalk = function(n, dim) {
  var prevStates = (n==2) ? [init(dim), init(dim)] : semiMarkovWalk(n-1, dim);
  var newState = transition(last(prevStates), secondLast(prevStates));
  return prevStates.concat([newState]);
};

var positions = semiMarkovWalk(80, 2);
