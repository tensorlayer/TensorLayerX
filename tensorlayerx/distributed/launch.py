import os
BACKEND = 'torch'


# Set backend based on TL_BACKEND.
if 'TL_BACKEND' in os.environ:
    backend = os.environ['TL_BACKEND']
    if backend:
        BACKEND = backend


def main(args=None):
    if BACKEND == 'torch':
        from torch.distributed.run import get_args_parser, run
        def parse_args(args):
            parser = get_args_parser()
            parser.add_argument(
                "--use_env",
                default=False,
                action="store_true",
                help="Use environment variable to pass "
                "'local rank'. For legacy reasons, the default value is False. "
                "If set to True, the script will not pass "
                "--local_rank as argument, and will instead set LOCAL_RANK.",
            )
            return parser.parse_args(args)
        args = parse_args(args)
        run(args)
    else:
        raise NotImplementedError("This backend:{} is not supported".format(BACKEND))
    

if __name__ == "__main__":
    main()