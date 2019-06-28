// HMM 1
var transition = function(s) {
  return s ? flip(0.7) : flip(0.3)
}

var observeState = function(s) {
  return s ? flip(0.9) : flip(0.1)
}

observeState(transition(true))


// HMM 2
var hmm = function(n) {
  var prev = (n==1) ? {states: [true], observations:[]} : hmm(n-1)
  var newState = transition(prev.states[prev.states.length-1])
  var newObs = observeState(newState)
  return {
    states: prev.states.concat([newState]),
    observations: prev.observations.concat([newObs])
  }
}

hmm(4)


// HMM 3 : Conditioning
// some true observations (the data we observe):
var trueObs = [false, false, false]

var model = function(){
  var r = hmm(3)
  factor(_.isEqual(r.observations, trueObs) ? 0 : -Infinity)
  return r.states
};

viz.table(Infer({ model }))

0	    1	    2	    3	    probability
true	false	false	false	0.8296918550634864
true	true	false	false	0.092187983895943
true	false	false	true	0.03950913595540412
true	false	true	false	0.01693248683803033
true	true	true	false	0.010243109321771443
true	false	true	true	0.004389903995044901
true	true	false	true	0.004389903995044901
true	true	true	true	0.0026556209352740735
