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
                           (at-robby ?from)
                           (not (at-robby ?to))
                           (not (robby-charged)))
       :effect (and  (at-robby ?to)
		             (not (at-robby ?from))
		             (robby-charged)))


   (:action pick
       :parameters (?obj ?room ?gripper)
       :precondition  (and  (ball ?obj) (not (heavy ?obj)) (room ?room) (gripper ?gripper)
			                (at ?obj ?room)
			                (at-robby ?room)
			                (free ?gripper)
			                (not (carry ?obj ?gripper)))
       :effect (and (carry ?obj ?gripper)
		            (not (at ?obj ?room))
		            (not (free ?gripper))))

    (:action pick_heavy
       :parameters (?obj ?room ?gripper)
       :precondition  (and  (ball ?obj) (heavy ?obj) (room ?room) (gripper ?gripper)
			                (at ?obj ?room)
			                (at-robby ?room)
			                (free ?gripper)
			                (robby-charged)
			                (not (carry ?obj ?gripper)))
       :effect (and (carry ?obj ?gripper)
		            (not (at ?obj ?room))
		            (not (free ?gripper))
		            (not (robby-charged))))


   (:action drop
       :parameters  (?obj  ?room ?gripper)
       :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
			                (carry ?obj ?gripper)
			                (at-robby ?room)
			                (not (free ?gripper))
			                (not (at ?obj ?room))
			                (robby-charged))
       :effect (and (at ?obj ?room)
		            (free ?gripper)
		            (not (carry ?obj ?gripper))
		            (not (robby-charged))))

)

