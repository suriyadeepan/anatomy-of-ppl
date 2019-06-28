// language: javascript

///fold:
var Bernoulli = function(params) {
  return new dists.Bernoulli(params);
}

var cpsBinomial = function(k){
  _sample(
    function(a){
      _sample(
        function(b){
          _sample(
            function(c){
              k(a + b + c);
            },
            Bernoulli({ p: 0.5 }))
        },
        Bernoulli({ p: 0.5 }))
    },
    Bernoulli({ p: 0.5 }))
}
///

var unexploredFutures = []

function _sample(cont, dist) {
  var sup = dist.support()
  sup.forEach(function(s){unexploredFutures.push(function(){cont(s)})})
  unexploredFutures.pop()()
}

var returnVals = []

function exit(val) {
  returnVals.push(val)
  if (unexploredFutures.length > 0) {
    var next = unexploredFutures.pop()
    next()
  }
}

function Explore(cpsComp) {
  cpsComp(exit)
  return returnVals
}

Explore(cpsBinomial)
