// Simple Model
var binomial = function(){
  var a = sample(Bernoulli({ p: 0.1 }))
  var b = sample(Bernoulli({ p: 0.9 }))
  var c = sample(Bernoulli({ p: 0.1 }))
  factor((a && b) ? 0 : -Infinity)
  return a + b + c
}

viz(Infer({ model: binomial }))


// Interleave Factor
var binomial = function(){
  var a = sample(Bernoulli({ p: 0.1 }))
  var b = sample(Bernoulli({ p: 0.9 }))
  factor((a && b) ? 0 : -Infinity)  
  var c = sample(Bernoulli({ p: 0.1 }))
  return a + b + c
}

viz(Infer({ model: binomial }))


// Decompose Factor
var binomial = function(){
  var a = sample(Bernoulli({ p: 0.1 }))
  factor(a ? 0 : -Infinity)
  var b = sample(Bernoulli({ p: 0.9 }))
  factor(b ? 0 : -Infinity)
  var c = sample(Bernoulli({ p: 0.1 }))
  return a + b + c
}

viz(Infer({ model: binomial, method: 'enumerate', maxExecutions: 4 }))
