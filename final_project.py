import csv


csv.field_size_limit(1000000000000)

plot_list = []
control_dict = {}

with open('movieplots.csv',) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        i = 0;
        for row in spamreader:
            x = (' '.join(row))
            x = x.split(',')
            #print(x)
            plot = ''
            if len(x) >= 3:
                for phrase in x[2:]:
                    plot += phrase
                if x[1] != "unknown":
                    plot_list.append(plot)
                    control_dict[plot] = x[1]
            
print(control_dict)