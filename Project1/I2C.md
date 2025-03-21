Problem 1:α)
Απάντηση: Απο την εκφώνηση έχω τα σημαντικά δεδομένα
Α)Τα χ1,χ2 είναι ανεξάρτητα άρα f(x1,x2) = f(x1)f(x2).
Β) Οι δύο υποθέσεις έχουν ίσα Priors P(H0) = P(H1) = 0.5
Έχω εν τέλη τον πίνακα,κατα bayes

|  | H0 | H1 |
|--------|----|----|
| Prior Probability | 0.5 |0.5 |
| P. D.F. | f0(x) | f1(x) |
| Correct Decision Cost | C00 | C11 |
| Incorrect Decision Cost | C01 | C10 |
Γνωρίζουμε απο την θεωρία(βλ. Lecture 5 pg 8) οτι το optimum bayes test(αυτό που ελαχιστοποιεί το μεσο κόστος) είναι το εξης likelihood ratio test
$$H_1 ,if\ [
\frac{f_1(x_1, x_2)}{f_0(x_1, x_2)} > \frac{P(H_0)C_{00} - P(H_0)C_{01}}{P(H_1)C_{10} - P(H_1)C_{11}}  
]\ || \ H_o, if\ [
\frac{f_1(x_1, x_2)}{f_0(x_1, x_2)} < \frac{P(H_0)C_{00} - P(H_0)C_{01}}{P(H_1)C_{10} - P(H_1)C_{11}}  
]$$
Το οποίο καθώς (Α) και (Β) Γίνεται
$$H_1 ,if\ [
\frac{f_1(x_1) \cdot f(x_2)}{f_0(x_1) \cdot f_0(x_2)} > \frac{C_{00} - C_{01}}{C_{10} - C_{11}}
]\ ||\ H_o,if\ [
\frac{f_1(x_1) \cdot f(x_2)}{f_0(x_1) \cdot f_0(x_2)} < \frac{C_{00} - C_{01}}{C_{10} - C_{11}}
]$$

Επίσης γνωρίζω( Lecture 5 pg 8) οτι 
Average cost == Error probability αν :
$$[
 \frac{C_{00} - C_{01}}{C_{10} - C_{11}} = 1
]$$
Δηλαδή
|  | H0 | H1 |
|--------|----|----|
| Prior Probability | 0.5 |0.5 |
| P. D.F. | f0(x) | f1(x) |
| Correct Decision Cost | 0 | 0 |
| Incorrect Decision Cost | 1 | 1 |
(Ίσο κόστος για οποιοδήποτε λάθος)

Αρα τελικά 
$$H_1 ,if\ [
\frac{f_1(x_1) \cdot f(x_2)}{f_0(x_1) \cdot f_0(x_2)} > 1
] \ || H_o,if\ [
\frac{f_1(x_1) \cdot f(x_2)}{f_0(x_1) \cdot f_0(x_2)} < 1
]$$

Επίσης έχω:
Ηο :f₀(x) ~ N(0,1) ,αρα:
$$f_0(x) = \frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}$$
Ομοίως H1  : f₁(x) ~ 0.5[N(-1,1) + N(1,1)], αρα:
$$f_1(x) = \frac{1}{2}\frac{1}{\sqrt{2\pi}}(e^{-\frac{(x+1)^2}{2}} + e^{-\frac{(x-1)^2}{2}})$$


$$\frac{f_1(x)}{f_0(x)} = \frac{\frac{1}{2}(e^{-\frac{(x+1)^2}{2}} + e^{-\frac{(x-1)^2}{2}})}{e^{-\frac{x^2}{2}}}$$
Και απο (Α)
$$\frac{f_1(x_1,x_2)}{f_0(x_1,x_2)} = \frac{f_1(x_1)}{f_0(x_1)} \cdot \frac{f_1(x_2)}{f_0(x_2)}$$
Εν τέλη προκείπτει με αντικατάσταση η εξής σχέση για το τέστ
$$\frac{1}{2}\left(e^{-\frac{(x_1 + 1)^2}{2}} + e^{-\frac{(x_1 - 1)^2}{2}}\right) \cdot \frac{1}{2}\left(e^{-\frac{(x_2 + 1)^2}{2}} + e^{-\frac{(x_2 - 1)^2}{2}}\right) > e^{-\frac{x_1^2}{2}} e^{-\frac{x_2^2}{2}}$$
Την οποία θα απλοποιήσω
Παρατηρώ
$$( e^{-\frac{(x_i \pm 1)^2}{2}} = e^{-\frac{x_i^2}{2}} \cdot e^{\mp x_i} \cdot e^{-\frac{1}{2}} )$$

Και έτσι απλοποιώ :
$$[
\frac{1}{4} \left(e^{-\frac{x_1^2}{2}}(e^{x_1} + e^{-x_1})\right) \left(e^{-\frac{x_2^2}{2}}(e^{x_2} + e^{-x_2})\right) > e^{-\frac{x_1^2}{2}} e^{-\frac{x_2^2}{2}}
]$$
Αφου όμως
$$(e^{x} + e^{-x} = 2 \cosh(x))$$
Τελικά έχω την απλή εκφραση 
$$\cosh(x_1) \cosh(x_2) > 1$$
Άρα η τελική σχέση είναι:
$$H_1 ,if\ [
\cosh(x_1) \cosh(x_2) > 1
] \ || H_o,if\ [
\cosh(x_1) \cosh(x_2) < 1
]$$








