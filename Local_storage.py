You can use the `st.session_state` to track whether the page has been refreshed or not. Here's an example:

```
import streamlit as st

# Initialize session state
if 'page_refreshed' not in st.session_state:
    st.session_state.page_refreshed = False

# Check if page has been refreshed
if st.session_state.page_refreshed:
    st.write("Page has been refreshed")
else:
    st.write("Page has not been refreshed")
    st.session_state.page_refreshed = True
```

In this example, we initialize a `page_refreshed` variable in the `session_state` to `False`. Then, we check if the page has been refreshed by checking the value of `page_refreshed`. If it's `True`, we display a message indicating that the page has been refreshed. If it's `False`, we display a message indicating that the page has not been refreshed and set `page_refreshed` to `True`.

However, this approach has a limitation. The `session_state` is reset when the user closes the browser or clears the browser's cache.

To overcome this limitation, you can use a more robust approach, such as using a backend database or a cookie-based solution.

# Using Cookies
Streamlit provides a `st.experimental_get_query_params` and `st.experimental_set_query_params` function to get and set URL query parameters. However, these functions do not provide a way to store data locally on the client-side.

One workaround is to use the `st.experimental_show_shortcode` function to display a shortcode that sets a cookie. However, this approach requires some JavaScript code.

# Using Local Storage
Another approach is to use the browser's local storage to store a flag indicating whether the page has been refreshed. You can use the `st.experimental_show_shortcode` function to display a shortcode that sets a local storage item.

Here is an example using local storage:

```
// Set local storage item
localStorage.setItem('pageRefreshed', 'true');
```

```
import streamlit as st

# Check if local storage item is set
st.write("<script>console.log(localStorage.getItem('pageRefreshed'));</script>", unsafe_allow_html=True)

if st.button("Check Refresh"):
    st.write("<script>if (localStorage.getItem('pageRefreshed') === 'true') {console.log('Page has been refreshed');} else {console.log('Page has not been refreshed'); localStorage.setItem('pageRefreshed', 'true');}</script>", unsafe_allow_html=True)
```

In this example, we use JavaScript to set a local storage item `pageRefreshed` to `'true'`. Then, we use Python to check if the local storage item is set. If it is set, we display a message indicating that the page has been refreshed. If it is not set, we display a message indicating that the page has not been refreshed and set the local storage item to `'true'`.
