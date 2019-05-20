class Template(object):
    def __init__(self, name, code):
        self.name = name
        self.code = code.strip()

    def compile(self, ctx):
        compiled = self.code
        for key, val in ctx.items():
            compiled = compiled.replace("{%s}" % key, str(val))
        return compiled
