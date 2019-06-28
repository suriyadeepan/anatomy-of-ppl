// language: javascript

///fold:
var Bernoulli = function(params) {
  return new dists.Bernoulli(params);
}

function cpsBinomial(k){
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
var currScore = 0

function _sample(cont, dist, params) {
  var sup = dist.support(params)
  sup.forEach(function(s){
  var newscore = currScore + dist.score(s);
  unexploredFutures.push({k: function(){cont(s)}, score: newscore})})
  runNext()
}

function runNext(){
  var next = unexploredFutures.pop()
  currScore = next.score
  next.k()}

var returnHist = {}

function exit(val) {
  returnHist[val] = (returnHist[val] || 0) + Math.exp(currScore)
  if( unexploredFutures.length > 0 ) {runNext()}
}

function ExploreWeighted(cpsComp) {
  cpsComp(exit)
  return returnHist
}

ExploreWeighted(cpsBinomial)
