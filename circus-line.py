import os, time, sys

header = '''
##############################################################
#####          Welcome to the SpyKING CIRCUS             #####
#####               (watching daemon)                    #####
#####          Written by P.Yger and O.Marre             #####
##############################################################
'''

print header

if len(sys.argv) < 2:
    queue_file = 'tasks.todo'
    print "No watch file specified, used default"
else:
    queue_file = sys.argv[-1]

print "Watch file :", queue_file

while True:
    tasks =  open(queue_file, 'r')
    lines = tasks.readlines()
    tasks.close()
    if len(lines) > 0:
        task = lines[0]
        print "Processing %s..." %task
        os.system('python spyking-circus.py %s' %task)
        # We reload the file that may have been changed..
        tasks =  open(queue_file, 'r')
        lines = tasks.readlines()
        tasks.close()
        # And we remove its first line...
        tasks = open(queue_file, 'w')
        for line in lines[1:]:
            tasks.write(line)
        tasks.close()
    else:
        time.sleep(60)
        sys.stdout.write('\r'+ "Waiting for new tasks in %s...." %queue_file)
