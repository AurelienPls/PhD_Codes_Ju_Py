import json
import urllib.request


# net agent
# ---------

def call_uri(uri, params = {}, method = 'POSTJSON', return_req = False, 
             ssluri = None):
    """
    _Note:_ if return_req is True, also returns the object returned by
    urlopen() (useful to retrieve the effective uri (in case of a redirect),
    the error code etc...): (data, urlopen_ret)

    """

    import collections

    def _is_https(uri):
        from urllib.parse import urlparse
        return urlparse(uri).scheme == 'https'

    if method == "POST":
        req = urllib.request.Request(uri, post_data, headers)

    elif method == "POSTJSON":
        # method exists to distinguish plain POST (multiparts) and POSTING a
        # json document

        # convert json_dict to JSON
        json_data = json.dumps(params)

        # convert str to bytes (ensure encoding is OK)
        post_data = json_data.encode('utf-8')
    
        # we should also say the JSON content type header
        headers = {}
        headers['Content-Type'] = 'application/json'

        # now do the request for a url
        req = urllib.request.Request(uri,
                                     post_data,
                                     headers)

    elif method == "GET":

        if params:
            # 20211220: params may be 1. dict, or 2. list of k,v (allowing for
            # several occurence of the same parameter name to emulate "lists"
            # in query string
            # turn None to '', otherwise, None will be encoded 'None'...
            if isinstance(params, dict):
                params = list(params.items())

            params = [(k, v) if v is not None else '' for k, v in params]

            # url encode params
            query_str = urllib.parse.urlencode(params, doseq = True)
            uri = uri + "?{}".format(query_str)
    
        # now do the request for a url
        req = urllib.request.Request(uri)

    # send the request
    result = None # error
    urlopen_ret = None

    if ssluri == True or (ssluri == None and _is_https(uri)):
        import ssl
        context = ssl.create_default_context()
        uopen = lambda: urllib.request.urlopen(req, context = context)
    else:
        uopen = lambda: urllib.request.urlopen(req)

    with uopen() as f_res:
        urlopen_ret = f_res
        json_data = f_res.read().decode('utf-8')
        try: # is it json document ? -> test header no ?
            result = json.loads(json_data,
                                object_pairs_hook = collections.OrderedDict)
        except Exception as e:
            result = json_data

    if return_req:
        return (result, urlopen_ret)
    else:
        return result



### streaming of paginated services

def paginated_stream_proxy(func, page_size = 20, headers = 0):
    """
    Proxies a paginated function as a (possibly infinite) stream.

    A stream function is a function that returns a sub-list from a
    larger one and takes as argument the start position and number of
    elements of the sublist relatively to the parent one.

    `func` is func(page_size, offset)

    
    `headers` is the number of rows to skip to find the first data row in
    a page stream. The header will be returned as the first rows of
    the stream.

    _Note:_ the maximum cache size is equal to the value of `page_size`.
    """

    class paginated_stream(object):
        def __init__(self, func, page_size):
            self.cache = []
            self.cache_position = 0

            self.last_page = False
            
            self.func = func

            self.offset = 0
            self.page_size = page_size
            
        def __iter__(self):
            return self

        def __next__(self):

            if (not self.cache
                or (self.cache_position == len(self.cache)
                    and not self.last_page)):
                
                update = self.func(self.page_size, self.offset)
                h = None
                # does the result have headers that we want to skip ?
                if headers > 0:
                    h = update[0:headers]
                    update = update[headers:]
                    
                if not update:
                    raise StopIteration

                if len(update) < self.page_size:
                    self.last_page = True

                if self.offset == 0 and h:
                    self.cache = h + update
                else:
                    self.cache = update
                    
                self.cache_position = 0
                
                self.offset += self.page_size

            try:
                next_element = self.cache[self.cache_position]
                self.cache_position += 1
            except IndexError:
                raise StopIteration
                
            return next_element

    return paginated_stream(func, page_size)

    
def stream_proxy_builder(f, page_size = 20, headers = 0):
    def _build_stream():
        return paginated_stream_proxy(f, page_size, headers)
    return _build_stream



### Bottle.py session

def get_session(cf, request, response):
    k = cf['cookie_key']    
    sc = request.get_cookie(cf['session_cookie_name'], secret = k)
    if sc:
        session = json.loads(sc)
        return session
    
    return {}


def set_session(cf, request, response, s):
    k = cf['cookie_key']    
    sc = response.set_cookie(cf['session_cookie_name'],
                             json.dumps(s),
                             secret = k)

    
def delete_cookie(request, response, cookie_name, cookie_path = '/'):
    cookie = request.get_cookie(cookie_name)
    response.delete_cookie(cookie_name, path = cookie_path)
    return cookie



