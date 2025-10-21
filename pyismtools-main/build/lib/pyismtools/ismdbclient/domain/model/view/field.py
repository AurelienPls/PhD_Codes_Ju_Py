import pyismtools.utils2.domain.model2 as m

class Field(m.Model):
    pass

def create(name, id = None, unit = '', utype = '', label = ''): 
    _field = Field({ 
                    'id':id,
                    'name': name,
                    'unit': unit,
                    'utype': utype,
                    'label': label
                    })

    return _field

def utype(field, utype_ = None):
    if utype_ == None:
        return m.get(field, 'utype')

    m.set_(field, {'utype': utype_})
    return utype_
