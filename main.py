from module1 import InteractionModule
from module2 import ParserModule
from module3 import DummyInfoModule  # keep if you still use it
from module4 import BasicTaskPerformer
from module5 import ContentWriterClient
from module6 import Overseer


# Initialize modules
m1 = InteractionModule()
m2 = ParserModule()
m3 = DummyInfoModule()          # Optional: remove if not needed
m4 = BasicTaskPerformer()       # Real executor
m5 = ContentWriterClient()      # Real writer


# Create overseer
overseer = Overseer(
    module1=m1,
    module2=m2,
    module3=m3,
    module4=m4,
    module5=m5
)


# Run system
overseer.run("Write an email to my physics teacher with my coursework.")