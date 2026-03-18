(define (domain lights)
    (:requirements :strips :conditional-effects)

    (:predicates
        (light_on ?b)
        (light_out ?b)
        (adjacent ?b1 ?b2)
    )

    (:action press_button
        :parameters (?b)
        :precondition ()
        :effect (and
            (forall (?b2)
                    (when (and (adjacent ?b ?b2) (light_on ?b2))
                          (and (light_out ?b2) (not (light_on ?b2)))
                    )
            )
            (forall (?b2)
                    (when (and (adjacent ?b ?b2) (light_out ?b2))
                          (and (light_on ?b2) (not (light_out ?b2)))
                    )
            )
            (when (light_out ?b) (and (light_on ?b) (not (light_out ?b))))
            (when (light_on ?b) (and (light_out ?b) (not (light_on ?b))))

        )
    )
)