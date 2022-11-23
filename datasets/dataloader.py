
class ConcatDataLoader():
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __iter__(self):
        self.source_iter = iter(self.source)
        self.target_iter = iter(self.target)

        return self
    
    def __next__(self):

        try:
            t_imgs, t_label = next(self.target_iter)
        except StopIteration:
            self.target_iter = iter(self.target)
            t_imgs, t_label = next(self.target_iter)
        
        try:
            s_imgs, s_label = next(self.source_iter)
        except StopIteration:
            raise StopIteration
        return (s_imgs, s_label), (t_imgs, t_label)

    def __len__(self):
        return max(len(self.source), len(self.target))