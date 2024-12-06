from dataclasses import dataclass, field


config_dataclass = dataclass(frozen=True, repr=False)

make_required_field = lambda: field(default=None, metadata={"required": True})

make_fixed_field = lambda default: field(default=default, metadata={"fixed": True})

make_factory_field = lambda factory: field(default_factory=factory)


@config_dataclass
class BaseConfig:
    """Basic Configurations Dataclass
        NOTE: some parameters are tagged as "required" like `attr: Type = make_required_field()`, \
            indicating they MUST be set to some values except `None` during initialization,
        while some parameters are tagged as "fixed" like `attr: Type = make_fixed_field(default_val)`, \
            indicating they can NOT be set during initialization and remain their own default values.
    """
    
    def __post_init__(self):
        """Post-initialization method for BaseConfig, \
            ensuring all required fields are set and no fixed fields are modified.
        """
        missing_fields = []
        modified_fixed_fields = []
        
        for field_name, field_def in self.__dataclass_fields__.items():
            if field_def.metadata.get("required", False) and field_def.metadata.get("fixed", False):
                raise AttributeError(f"Field {field_name} cannot have set both 'required' and 'fixed' metadata to `True` at the same time.")
            if field_def.metadata.get("required", False) and getattr(self, field_name) is None:
                missing_fields.append(field_name)
            if field_def.metadata.get("fixed", False) and getattr(self, field_name) != field_def.default:
                modified_fixed_fields.append(field_name)
        
        if missing_fields or modified_fixed_fields:
            error_msg = "BaseConfig initialization failed due to: \n"
            if missing_fields:
                error_msg += f"Missing required fields: {', '.join(missing_fields)}\n"
            if modified_fixed_fields:
                error_msg += f"Modified fixed fields: {', '.join(modified_fixed_fields)}\n"
            
            raise ValueError(error_msg)

    def __repr__(self) -> str:
        """Customized __repr__ method for BaseConfig, \
            displaying all fields with their values in alphabetical order.
        """
        class_name = self.__class__.__name__
        repr_str = f"{'*'*20}   {class_name}   {'*'*20}\n"
        title_len = len(repr_str) - 1
        
        field_names = sorted(self.__dataclass_fields__.keys())
        for field_name in field_names:
            field_value = getattr(self, field_name)
            if isinstance(field_value, str):
                field_value = repr(field_value)
            repr_str += f"{field_name}: {field_value}\n"
        
        repr_str += f"{'*' * title_len}\n"
        
        return repr_str