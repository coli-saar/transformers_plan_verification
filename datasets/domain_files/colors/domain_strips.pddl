(define (domain color-bags)
 (:requirements :strips)

 (:predicates (color ?c)
              (bag ?b)
              (has-color ?b ?c)
 )

 (:action remove_all_color
    :parameters (?b ?c)
    :precondition (and (bag ?b) (color ?c))
    :effect (and (not (has-color ?b ?c))))

 (:action add_color
    :parameters (?b ?c)
    :precondition (and (bag ?b) (color ?c))
    :effect (and (has-color ?b ?c)))

)
