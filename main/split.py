SPLIT = dict(
    
    NEU_DET=dict(
        ALL_CLASSES_SPLIT1=("crazing","inclusion","patches",
                            "pitted_surface","rolled-in_scale","scratches"),
        BASE_CLASSES_SPLIT1=("crazing","inclusion","patches",),
        NOVEL_CLASSES_SPLIT1=("pitted_surface","rolled-in_scale","scratches"),
        
        ALL_CLASSES_SPLIT2=("inclusion","rolled-in_scale","scratches",
                            "crazing","patches","pitted_surface"),
        BASE_CLASSES_SPLIT2=("inclusion","rolled-in_scale","scratches"),
        NOVEL_CLASSES_SPLIT2=("crazing","patches","pitted_surface"),

        ALL_CLASSES_SPLIT3=("pitted_surface","patches","scratches",
                            "crazing","inclusion","rolled-in_scale"),
        BASE_CLASSES_SPLIT3=("pitted_surface","patches","scratches"),
        NOVEL_CLASSES_SPLIT3=("crazing","inclusion","rolled-in_scale")
        ),
    
    DeepPCB=dict(
        ALL_CLASSES_SPLIT1=("open_circuit","short","mouse_bite",
                            "spur","spurious_copper","pin_hole"),
        BASE_CLASSES_SPLIT1=("open_circuit","short","mouse_bite"),
        NOVEL_CLASSES_SPLIT1=("spur","spurious_copper","pin_hole"),
        
        ALL_CLASSES_SPLIT2=("mouse_bite","spur","pin_hole",
                            "open_circuit","short","spurious_copper"),
        BASE_CLASSES_SPLIT2=("mouse_bite","spur","pin_hole"),
        NOVEL_CLASSES_SPLIT2=("open_circuit","short","spurious_copper"),

        ALL_CLASSES_SPLIT3=("open_circuit","mouse_bite","pin_hole",
                            "short","spur","spurious_copper"),
        BASE_CLASSES_SPLIT3=("open_circuit","mouse_bite","pin_hole"),
        NOVEL_CLASSES_SPLIT3=("short","spur","spurious_copper")
        )
    )
