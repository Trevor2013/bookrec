import threading
from typing import Any, Optional

__all__ = ['BaseRequest', 'TornadoRequest', 'HttpRequest', 'http_request']


class BaseRequest:
    def __init__(self, request):
        self._request = request

    def get_argument(self,
                     name: str,
                     default: Any = None) -> Optional[str]:
        raise NotImplementedError

    def get_arguments(self,
                      name: str,
                      default: list = None) -> Optional[list]:
        raise NotImplementedError

    def get_query_argument(self,
                           name: str,
                           default: Any = None) -> Optional[str]:
        raise NotImplementedError

    def get_query_arguments(self,
                            name: str,
                            default: list = None) -> Optional[list]:
        raise NotImplementedError

    def get_from_header(self, name: str, default: Any = None) -> Optional[str]:
        raise NotImplementedError

    def get_from_cookie(self, name: str, default: Any = None) -> Optional[str]:
        raise NotImplementedError

    def get_body(self) -> Optional[str]:
        raise NotImplementedError


class TornadoRequest(BaseRequest):
    def __init__(self, request):
        self.request = request
        super().__init__(request)

    def get_argument(self,
                     name: str,
                     default: Any = None) -> Optional[str]:
        return self.request.get_argument(name, default=default)

    def get_arguments(self,
                      name: str,
                      default: list = None) -> Optional[list]:
        return self.request.get_arguments(name) or default

    def get_query_argument(self,
                           name: str,
                           default: Any = None) -> Optional[str]:
        return self.request.get_query_argument(name, default=default)

    def get_query_arguments(self,
                            name: str,
                            default: list = None) -> Optional[list]:
        return self.request.get_query_arguments(name) or default

    def get_from_header(self,
                        name: str,
                        default: Any = None) -> Optional[dict]:
        return self.request.request.headers.get(name, default)

    def get_from_cookie(self,
                        name: str,
                        default: Any = None) -> Optional[str]:
        return self.request.request.get_cookie(name, default=default)

    def get_body(self) -> Optional[str]:
        return self.request.request.body


class HttpRequest:

    _instance_lock = threading.Lock()
    _request: BaseRequest = None

    @classmethod
    def configure(cls, *args, **kwargs):
        with HttpRequest._instance_lock:
            print('hhhhhh configure')
            if not hasattr(HttpRequest, '_instance'):
                HttpRequest._instance = HttpRequest(*args, **kwargs)
        return HttpRequest._instance

    def set_request_proxy(self, request: BaseRequest):
        '''
        Set request proxy.

        usage:
            HttpRequest.configure().set_request_proxy(TornadoRequest)

        :param request: <BaseRequest>
        :return:
        '''
        self._request = request

    @property
    def request(self):
        if not self._request:
            self._request = TornadoRequest
        return self._request


http_request = HttpRequest.configure()
