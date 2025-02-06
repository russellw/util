(define fact
  (lambda (n)
    (if n
        (* n (fact (- n 1)))
        1)))

(define loop
  (lambda (n)
    (if n
        ((lambda (ignored)
           (loop (- n 1)))
         (fact 10))
        0)))

(fact 6)
