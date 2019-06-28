def factorial(n):
  if n == 0:
    return 1

  return factorial(n - 1) * n


def factorial_tail_recursiion(n, a=1):
  if n == 0:
    return a
  else:
    return factorial_tail_recursiion(n - 1, a * n)


def factorial_tail_recursiion_cps(k, n, a=1):
  if n == 0:
    return k(a)
  else:
    return factorial_tail_recursiion_cps(k, n - 1, a * n)


def factorial_cps(k, n):

  if n == 0:
    k(1)
  else:
    factorial_cps(
        lambda x : k(n * x),
        n - 1
        )


def error_handling_cps(k, err, n):
  if n < 0:
    err('n < 0')
  elif n == 0:
    k(1)
  else:
    error_handling_cps(
        lambda x : k(n * x),
        err,
        n - 1
        )


def err(s):
  print('Error : ', s)


if __name__ == '__main__':
  # factorial_cps(print, 5)
  # print(factorial_tail_recursiion(5))
  # factorial_tail_recursiion_cps(print, 5)
  error_handling_cps(print, err, -1)
