# **Project Idea Title 2: Automated design using evolutionary computation**

## **What problem is this project solving?**
The problem is how to automate the design of complex, possibly moving structures. For example,
moving robots or neural networks.
This project involves the creation of a system which uses evolutionary computing or artificial life
techniques to carry out a directed exploration of a space of possibilities. It is up to the student to
define what the space represents – the example seen in the AI course was the space of possible
forms for 3D creatures. The method of exploration is also up to the student, again, the example seen
in the AI course was to direct the exploration towards creatures with desirable characteristics. A
successful project will contain a functional system with a well defined space and a well defined
method for exploring that space.

## **What is the background and context to the question above in 150 words or fewer?**
Statistical machine learning techniques currently dominate in many problem domains such as image
analysis and speech analysis. However, there is another class of problem where the technique of
learning statistical patterns in large datasets cannot necessarily be applied. Evolutionary
computation and artificial life techniques can be applied to problems where there is minimal data
available and no pre-existing correct and incorrect solutions.

## **Recommended sources to begin research**
+ Lehman, Joel, et al. "The surprising creativity of digital evolution: A collection of anecdotes
from the evolutionary computation and artificial life research communities." Artificial life 26.2 (2020): 274-306.
  + an introduction to a range of systems that evolve forms
+ Sims, Karl. "Evolving virtual creatures." Proceedings of the 21st annual conference on Computer graphics and interactive techniques. 1994.
  + classic paper evolving moving creatures
+ Yee-King, Matthew John. "The use of interactive genetic algorithms in sound design: a comparative study." Computers In Entertainment 14.3 (2016).
  + evolving sound synthesis circuits
+ Thompson, Adrian. "An evolved circuit, intrinsic in silicon, entwined with physics."International Conference on Evolvable Systems. Springer, Berlin, Heidelberg, 1996.
  + classic paper about evolving circuits -might be possible with a cheap, modern day

FPGA, but challenging!

## **What would the final product look like?**
(e.g. presentation, usability, functionality, results)?
Presentation: We would expect the student to present in various media, detailed information about
the following:
+ Review of related work
+ What is the problem the system is addressing?
+ How has the solution space been represented parametrically – how are the solution
represented at a data level and how are they expressed such that they can be tested?
+ How does the algorithm work?
+ How well does the system perform? Do the solutions improve over the runtime of the
algorithm? * How did the student overcome challenges and make changes to the encoding
scheme, testing environment and so on to improve the system’s performance?

## **What would a prototype look like?**
What would it show?
What does it need to prove?
What IS important to make clear?
What is NOT important at this stage?
We recommend that at the prototype stage, the student has specified and ideally, implemented the
encoding scheme. So it should be clear what the problem is (e.g. designing robots, designing
buildings etc.) and how the system encodes the problem space (e.g. genetic encoding scheme). The
student should also have a clear and viable technical plan for how the potential solutions can be
evaluated (e.g. running robots in simulation).

## **What kinds of techniques/processes are relevant to this project?**
+ Review of relevant literature and description of the problem domain
+ Encoding solutions in a manner that is appropriate for the application of evolutionary
computing
+ Designing a fitness function suitable for measuring the performance
+ Using a genetic algorithm to iteratively evolve a population of solutions to a well specified
problem
+ Ensuring the evaluation of solutions is efficient enough to allow the evaluation of thousands
of solutions in a reasonable time
+ Evaluating the performance of the system as a whole and measuring the effect of different
settings

## **What would the output of these techniques/processes look like?**
+ Review of relevant literature and description of the problem domain
  + Section in the report describing similar work in the literature and describing the
problem domain
+ Encoding solutions in a manner that is appropriate for the application of evolutionary
computing
  + Genetic encoding scheme clearly described in the report and implemented in code
+ Designing a fitness function suitable for measuring the performance
  + Fitness function(s) clearly described and implemented in code
+ Using a genetic algorithm to iteratively evolve a population of solutions to a well specified
problem
  + Genetic algorithm implemented (can use a pre-made library if that helps), genetic
operators, selection and breeding working
+ Ensuring the evaluation of solutions is efficient enough to allow the evaluation of thousands
of solutions in a reasonable time
  + Measurements of how long it takes to evaluate solutions and demonstrating that
this allows meaningful evolution.
+ Evaluating the performance of the system as a whole and measuring the effect of different
settings
  + Evidence presented in the report of multiple runs of the system showing variation in
performance between runs. A comparison of runs with different parameter settings.
How will this project be evaluated and assessed by the student (i.e. during iteration of the
project)?

## **What criteria are important?**
+ Does the encoding scheme work?
+ Does the fitness function work?
+ Does the overall algorithm work, i.e. do the solutions improve as the algorithm proceeds?
+ How do features of the encoding scheme, fitness function and evolutionary algorithm
impact on the performance? Can you compare performance with different features enabled
or disabled?
+ Is the code well organised and well commented?
+ Is it absolutely clear which code has been written by the student and which has not?
+ Are the descriptions in the report sufficient for a tutor to understand how the code works
and how the system has been evaluated?


---

## **Mark Scheme**

### **For this brief, what would a minimum pass (e.g. 3rd) student project look like?**
+ Brief but limited review of the literature
+ Working but simplistic encoding scheme with examples of genetic data and how that is
converted into solutions
+ Working fitness function
+ Limited evidence of evaluation
+ Limited but complete report

### **For this brief, what would a good (e.g. 2:2 – 2:1) student project look like?**
+ Review of the literature which shows evidence of wide reading
+ Working encoding scheme with examples of genetic data and how that is converted into
solutions
+ Working fitness function
+ Evidence of effort to optimise the performance of the system
+ Evidence of meaningful evaluation of different aspects of the encoding scheme, fitness
function etc.
+ Evidence of significant technical work on the part of the student
+ Complete, clearly written report

### **For this brief, what would an outstanding (e.g. 1st) student project look like?**
+ Extensive review of the literature which shows evidence of wide reading and critique of
previous work
+ Working encoding scheme with examples of genetic data and how that is converted into
solutions
+ Working fitness function with multiple elements
+ Evidence of successfully optimising the performance of the system
+ Evidence of extensive evaluation and further development of different aspects of the
encoding scheme, fitness function etc. involving challenging technical work by the student
+ Complete, clearly written report