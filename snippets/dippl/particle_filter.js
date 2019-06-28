// ----------------- Particle Filter --------------------- //

var Bernoulli = function(params) {
  return new dists.Bernoulli(params);
}

var _sample = function(k, dist){
  return k(dist.sample());
}

var copySample = function(s){
  return {
    value: s.value,
    score: 0,
    continuation: s.continuation
  }
}

var resample = function(samples){
  var weights = samples.map(
    function(sample){return Math.exp(sample.score);});
  var newSamples = [];
  for (var i=0; i<samples.length; i++){
    var j = dists.discreteSample(weights);
    newSamples.push(copySample(samples[j]));
  }
  return newSamples;
}

var cpsHmm = function(k, states, observations){
  var prevState = states[states.length - 1];
  _sample(
    function(state){
      _factor(
        function(){
          if (observations.length == 0) {
            return k(states);
          } else {
            return cpsHmm(k, states.concat([state]), observations.slice(1));
          }
        },
        (state == observations[0]) ? 0 : -1);
    },
    Bernoulli({p: prevState ? .9 : .1}));
}

var runCpsHmm = function(k){
  var observations = [true, true, true, true];
  var startState = false;
  return cpsHmm(k, [startState], observations);
}
///


var samples = [];
var sampleIndex = 0;


var _factor = function(k, score){
  samples[sampleIndex].score += score;
  samples[sampleIndex].continuation = k; // NEW

  if (sampleIndex < samples.length-1){
    sampleIndex += 1;
  } else {
    samples = resample(samples);
    sampleIndex = 0;
  }

  samples[sampleIndex].continuation();
}


var pfExit = function(value){

  // Store sampled value
  samples[sampleIndex].value = value;

  if (sampleIndex < samples.length-1){
    // If samples unfinished, resume computation for next sample
    sampleIndex += 1;
    samples[sampleIndex].continuation(); // NEW
  } else {
    samples.forEach(function(x){print(JSON.stringify(x));});
  }
};


var SimpleParticleFilter = function(cpsComp, numSamples){

  // Create placeholders for samples
  for (var i=0; i<numSamples; i++) {
    var sample = {
      value: undefined,
      score: 0,
      continuation: function(){cpsComp(pfExit)} // NEW
    };
    samples.push(sample);
  }

  // Run computation from beginning
  samples[sampleIndex].continuation();
};


SimpleParticleFilter(runCpsHmm, 20);
