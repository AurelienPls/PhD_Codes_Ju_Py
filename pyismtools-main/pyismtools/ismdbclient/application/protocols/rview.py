from typing import Protocol

DEFAULT_ASEARCH_LIMIT = 5
DEFAULT_CUTOUT_PAGE_SIZE = 100
DEFAULT_CUTOUT_HEADERS = 2 # number of rows used as header in a cutout result.

class Rview(Protocol):
    def field_meta(self, field):
        pass

    def fields_asearch(self, filters, 
                       limit = DEFAULT_ASEARCH_LIMIT, offset = 0, cache = None):
        pass

    def sync_cutout(self, select, where, 
                    limit = DEFAULT_CUTOUT_PAGE_SIZE, 
                    offset = 0, 
                    inc_entity = True):
        pass

    def sync_cutout_stream(self, select, where, inc_entity = True,
                           page_size = DEFAULT_CUTOUT_PAGE_SIZE,
                           headers = DEFAULT_CUTOUT_HEADERS):
        pass

    def entity(self, id, select = []):
        pass

    def schema(self):
        pass

