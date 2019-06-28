// HMM
var hmmRecur = function(n, states, observations){
  var newState = transition(states[states.length-1]);
  var newObs = observeState(newState);
  var newStates = states.concat([newState]);
  var newObservations = observations.concat([newObs]);
  return (n==1) ? { states: newStates, observations: newObservations } : 
  hmmRecur(n-1, newStates, newObservations);
}

var hmm = function(n) {
  return hmmRecur(n,[true],[])
}

var trueObs = [false, false, false]

var model = function(){
  var r = hmm(3)
  factor(_.isEqual(r.observations, trueObs) ? 0 : -Infinity)
  return r.states
};

viz.table(Infer({ model }))


// PCFG
var pcfg = function(symbol, yieldsofar, trueyield) {
  if (preTerminal(symbol)){
    var t = terminal(symbol)
    if (yieldsofar.length < trueyield.length){
      factor(t==trueyield[yieldsofar.length] ?0:-Infinity)
    }
    return yieldsofar.concat([t])
  } else {
    return expand(pcfgTransition(symbol), yieldsofar, trueyield) }
}

var expand = function(symbols, yieldsofar, trueyield) {
  return symbols.length==0 ? yieldsofar : expand(symbols.slice(1), pcfg(symbols[0], yieldsofar, trueyield), trueyield)
}

var model = function(){
  var y = pcfg('start', [], ['tall', 'John'])
  return y[2]?y[2]:"" //distribution on next word?
}

viz.table(Infer({ model, method: 'enumerate', maxExecutions: 20}))
