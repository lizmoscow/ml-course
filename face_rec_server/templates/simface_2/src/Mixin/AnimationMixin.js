var AnimationMixin = {
    hideClass:          'is-invisible',
    hideAnimationClass: 'fadeOutRight',
    showAnimationClass: 'slideInRight',
    
    hide: function(animation, endCallback) {
        endCallback = endCallback || function(){};

        if(animation) {
            var $this = this;
            this.runAnimation(animation, function() {
                $this.$el.addClass($this.hideClass);
                endCallback();
            });
        } else {
            this.$el.addClass(this.hideClass);   
        }
    },
    
    show: function(animation, endCallback) {
        endCallback = endCallback || function(){};

        if(animation) {
            this.$el.removeClass(this.hideClass);
            this.runAnimation(animation, function() {
                endCallback();
            });
        } else {
            this.$el.removeClass(this.hideClass);   
        }
    },
    
    animationShow: function(endCallback) {
        this.show(this.showAnimationClass, endCallback);
    },

    animationHide: function(endCallback) {
        this.hide(this.hideAnimationClass, endCallback);
    }
};

Object.assign(AnimationMixin, AnimatableMixin);