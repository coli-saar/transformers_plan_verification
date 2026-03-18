(define (domain gripper-strips)
(:requirements :negative-preconditions)

   (:predicates (room ?r)
		(ball ?b)
		(gripper ?g)
		(at-robby ?r)
		(at ?b ?r)
		(free ?g)
		(carry ?o ?g)
		(heavy ?b)
		(robby-charged))


   (:action move
       :parameters  (?from ?to)
       :precondition (and  (room ?from) (room ?to)
                           (at-robby ?from))
       :effect (and  (at-robby ?to)
		             (robby-charged)))


   (:action pick
       :parameters (?obj ?room ?gripper)
       :precondition  (and  (ball ?obj) (not (heavy ?obj)) (room ?room) (gripper ?gripper)
			                (at ?obj ?room)
			                (at-robby ?room)
			                (free ?gripper))
       :effect (and (carry ?obj ?gripper)))

    (:action pick_heavy
       :parameters (?obj ?room ?gripper)
       :precondition  (and  (ball ?obj) (heavy ?obj) (room ?room) (gripper ?gripper)
			                (at ?obj ?room)
			                (at-robby ?room)
			                (free ?gripper)
			                (robby-charged))
       :effect (and (carry ?obj ?gripper)))


   (:action drop
       :parameters  (?obj  ?room ?gripper)
       :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
			                (carry ?obj ?gripper)
			                (at-robby ?room))
       :effect (and (at ?obj ?room)
		            (free ?gripper)))

)

