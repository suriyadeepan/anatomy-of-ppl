// Sample with Factor
var binomial = function(){
  var a = sampleWithFactor(Bernoulli({ p: 0.1 }), function(v){return v?0:-Infinity})
  var b = sampleWithFactor(Bernoulli({ p: 0.9 }), function(v){return v?0:-Infinity})
  var c = sample(Bernoulli({ p: 0.1 }))
  return a + b + c
}

viz(Infer({ model: binomial, method: 'enumerate', maxExecutions: 2 }))


// HMM
var hmmRecur = function(n, states, observations){
  var newState = transition(states[states.length-1])
  var newObs = sampleWithFactor(
    observeState(newState),
    function(v){return v==trueObs[observations.length] ? 0 : -Infinity})
  var newStates = states.concat([newState])
  var newObservations = observations.concat([newObs])
  return ((n==1) ? 
          { states: newStates, observations: newObservations } :
          hmmRecur(n-1, newStates, newObservations));
}

var hmm = function(n) {
  return hmmRecur(n,[true],[])
}

var model = function(){
  var r = hmm(3)
  return r.states
}

viz.table(Infer({ model, method: 'enumerate', maxExecutions: 500 }))


// Inserting Cancelling Heuristic Factors
var binomial = function(){
  var a = sample(Bernoulli({ p: 0.1 }))
  var b = sample(Bernoulli({ p: 0.9 }))
  var c = sample(Bernoulli({ p: 0.1 }))
  factor((a||b||c) ? 0 : -10)
  return a + b + c
}

viz(Infer({ model: binomial, method: 'enumerate', maxExecutions: 2}))


// ..
var binomial = function(){
  var a = sample(Bernoulli({ p: 0.1 }))
  factor(a ? 0 : -1)
  var b = sample(Bernoulli({ p: 0.9 }))
  factor(((a||b)?0:-1) - (a?0:-1))
  var c = sample(Bernoulli({ p: 0.1 }))
  factor(((a||b||c) ? 0:-10) - ((a||b)?0:-1))
  return a + b + c
}

viz(Infer({ model: binomial, method: 'enumerate', maxExecutions: 2 }))
