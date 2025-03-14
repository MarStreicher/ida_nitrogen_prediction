import argparse
from pydantic import BaseModel
from typing import Literal, Type

from models.base_model import BaseExperimentArgs
from registry import models

def str_to_bool(value):
    if value.lower() in ["true", "t"]:
        return True
    elif value.lower() in ["false", "f"]:
        return False
    elif value.lower() in ["none", "n"]:
        return None
    else:
        raise argparse.ArgumentTypeError("Invalid boolean value: {}".format(value))

def str_to_list(value):
    import json
    
    value = value.strip("'").strip('"')
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise argparse.ArgumentTypeError("Invalid list value: {}".format(value))
    return parsed

def _parser_from_model(parser: argparse.ArgumentParser, base_args: Type[BaseModel]):
    fields = base_args.model_fields
    
    for name, field in fields.items():
        def get_type_args():
            is_optional = getattr(field.annotation, "__name__", None) == "Optional"

            anno_args = getattr(field.annotation, "__args__", None)
            field_type = (
                anno_args[0]
                if anno_args is not None and is_optional
                else field.annotation
            )
            assert field_type is not None
            field_type_name = getattr(field_type, "__name__", None)
            is_literal = (
                field_type_name == "Literal"
                if is_optional
                else getattr(field.annotation, "__origin__", None) is Literal
            )
            is_bool = field_type_name == "bool"
            is_list = field_type_name == "list"

            if is_literal:
                return {"type": str, "choices": field_type.__args__}
            if is_bool:
                return {"type": str_to_bool}
            if is_list:
                return {"type": str_to_list}
            return {"type": field_type}
        
        parser.add_argument(
            f"--{name}", 
            dest = name,
            default = field.default,
            help = field.description,
            **get_type_args(),
        )
    
    return parser
        
def _create_arg_parser():
    base_parser = argparse.ArgumentParser(description='Process experiment arguments.')
    base_parser = _parser_from_model(base_parser, BaseExperimentArgs)
    base_args, unknown_base_args = base_parser.parse_known_args()
    
    assert{
        base_args.model in models
    }, f"{base_args.model} not found in registry."
    
    model_parser = argparse. ArgumentParser(description='Process model arguments.')
    model_parser = _parser_from_model(model_parser, models[base_args.model].get_args_model())
    
    return model_parser

def get_model_from_args():
    model_parser = _create_arg_parser()
    args, unknown_args = model_parser.parse_known_args()
    
    model_cls = models[args.model]
    config = model_cls.get_args_model()(**vars(args))
    model = model_cls(config)
    
    return config, model
    