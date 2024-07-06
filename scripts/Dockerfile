ARG INPUT_IMAGE_NAME

from ${INPUT_IMAGE_NAME} AS llm_custom

ARG EXP_DIR
WORKDIR $EXP_DIR

# Copy and install dependencies seperately first to avoid triggering
# a dependency reinstallation on every change in the code
# Copy using poetry.lock* in case it doesn't exist yet
COPY ./pyproject.toml ./poetry.lock* $EXP_DIR/
COPY ./src/lib_project/pyproject.toml ./src/lib_project/poetry.lock $EXP_DIR/src/lib_project/
COPY ./src/lib_vision/pyproject.toml ./src/lib_vision/poetry.lock $EXP_DIR/src/lib_vision/
COPY ./src/lib_dl_base/pyproject.toml ./src/lib_dl_base/poetry.lock $EXP_DIR/src/lib_dl_base/

# We create these files here to make poetry and pip happy
# and later copy them into the target image
RUN touch $EXP_DIR/README.md \
    && mkdir -p $EXP_DIR/src/ \
    && touch $EXP_DIR/src/__init__.py
ARG LIBRARIES="lib_project lib_vision lib_dl_base"
RUN for lib in $LIBRARIES; do \
        touch $EXP_DIR/src/$lib/README.md && \
        mkdir -p $EXP_DIR/src/$lib/$lib/ && \
        touch $EXP_DIR/src/$lib/$lib/__init__.py; \
    done

# TODO: using deepspeed stage 3 requires precompiling plugins.
# We should do this here, because the final image does not contain
# devel packages.
RUN poetry build --format wheel \
    && poetry export --without dev --with docker \
        --without-hashes --format constraints.txt --output constraints.txt
